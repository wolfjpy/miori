// LFM3-VL vision tower + projector, Metal + C++20.
//
// Layout of one image:
//   decoded RGB -> smart_resize (multiple of 32) -> antialiased bicubic
//   -> (x/255 - 0.5)/0.5 -> patchify in merge-block order (2x2 blocks, then
//   (dy, dx), channel-major, temporal frame duplicated) -> [N][1536] bf16
//   -> patchifier (1024) -> embedding_norm -> 14 layers (gated 3x3 depthwise
//   conv / bidirectional GQA with axial 2D RoPE) -> LayerNorm + 2-layer
//   projector -> one 2048-d language embedding per 2x2 block.
#include "vision.h"
#include "cpu_ref.h"
#include "ops.h"

#include <ImageIO/ImageIO.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace miori {

// From models/config.json vision_config.layer_types (1 = full attention).
static const int kVisionKind[14] = {0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

const char* Vision::layer_type(int i) {
    return (i >= 0 && i < 14 && kVisionKind[i]) ? "attention" : "conv";
}

// ---------------------------------------------------------------- image decode

static bool decode_rgb(const std::string& path, int& w, int& h, std::vector<uint8_t>& rgb,
                       std::string& err) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        nullptr, (const UInt8*)path.c_str(), (CFIndex)path.size(), false);
    if (!url) { err = "vision: bad image path: " + path; return false; }
    CGImageSourceRef src = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!src) { err = "vision: cannot open image: " + path; return false; }
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!img) { err = "vision: cannot decode image (unsupported format?): " + path; return false; }

    const size_t iw = CGImageGetWidth(img), ih = CGImageGetHeight(img);
    if (iw == 0 || ih == 0) { CGImageRelease(img); err = "vision: empty image"; return false; }
    std::vector<uint8_t> rgba(iw * ih * 4);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    const CGBitmapInfo kInfo = (CGBitmapInfo)((uint32_t)kCGImageAlphaNoneSkipLast |
                                              (uint32_t)kCGBitmapByteOrder32Big);
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), iw, ih, 8, iw * 4, cs, kInfo);
    CGColorSpaceRelease(cs);
    if (!ctx) { CGImageRelease(img); err = "vision: bitmap context failed"; return false; }
    CGContextSetBlendMode(ctx, kCGBlendModeCopy);
    // Default CTM (origin bottom-left) with a top-down backing store: the image
    // lands in memory top row first, which is what the patchifier wants.
    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)iw, (CGFloat)ih), img);
    CGContextRelease(ctx);
    CGImageRelease(img);

    w = (int)iw;
    h = (int)ih;
    rgb.resize(iw * ih * 3);
    for (size_t i = 0; i < iw * ih; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return true;
}

// ---------------------------------------------------------------- smart_resize

// Same rule as Lfm3VlImageProcessor.smart_resize, with factor = patch*merge = 32.
static void smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                         int& out_h, int& out_w) {
    if (height <= 0 || width <= 0) { out_h = factor; out_w = factor; return; }
    if ((double)std::max(height, width) / (double)std::min(height, width) > 200.0) {
        // The reference raises here; clamp instead so odd images still work.
        if (height > width) width = std::max(1, height / 200);
        else height = std::max(1, width / 200);
    }
    int h_bar = (int)std::llround((double)height / factor) * factor;
    int w_bar = (int)std::llround((double)width / factor) * factor;
    h_bar = std::max(h_bar, factor);
    w_bar = std::max(w_bar, factor);
    if ((int64_t)h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt((double)height * width / (double)max_pixels);
        h_bar = std::max(factor, (int)std::floor(height / beta / factor) * factor);
        w_bar = std::max(factor, (int)std::floor(width / beta / factor) * factor);
    } else if ((int64_t)h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt((double)min_pixels / ((double)height * width));
        h_bar = (int)std::ceil(height * beta / factor) * factor;
        w_bar = (int)std::ceil(width * beta / factor) * factor;
    }
    out_h = h_bar;
    out_w = w_bar;
}

// ---------------------------------------------------------------- bicubic resize

namespace {

struct AxisWeights {
    int start = 0;
    std::vector<float> w;
};

// Antialiased bicubic (a = -0.75) matching PIL/torchvision: when downscaling,
// the filter is widened by 1/scale and normalised, exactly like PIL's
// ImagingResample. torchvision's resize on uint8 is applied before rescale, so
// the result is rounded back to uint8 before normalisation.
std::vector<AxisWeights> bicubic_weights(int in, int out) {
    const double a = -0.75;
    auto filter = [&](double x) -> double {
        x = std::fabs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        return 0.0;
    };
    std::vector<AxisWeights> res((size_t)out);
    const double scale = (double)in / (double)out;
    double filterscale = scale;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = 2.0 * filterscale;
    const double ss = 1.0 / filterscale;
    for (int xx = 0; xx < out; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in) xmax = in;
        if (xmax <= xmin) xmax = std::min(in, xmin + 1);
        const int ksize = xmax - xmin;
        res[(size_t)xx].start = xmin;
        res[(size_t)xx].w.resize((size_t)ksize);
        double total = 0.0;
        for (int x = 0; x < ksize; ++x) {
            const double wv = filter((x + xmin - center + 0.5) * ss);
            res[(size_t)xx].w[(size_t)x] = (float)wv;
            total += wv;
        }
        if (total != 0.0) {
            for (auto& v : res[(size_t)xx].w) v = (float)((double)v / total);
        }
    }
    return res;
}

// RGB8 [sh][sw] -> RGB8 [dh][dw], separable antialiased bicubic.
bool resize_rgb(const std::vector<uint8_t>& src, int sw, int sh, int dw, int dh,
                std::vector<uint8_t>& dst, std::string& err) {
    const std::vector<AxisWeights> wx = bicubic_weights(sw, dw);
    const std::vector<AxisWeights> wy = bicubic_weights(sh, dh);
    std::vector<float> tmp((size_t)sh * dw * 3);
    for (int y = 0; y < sh; ++y) {
        const uint8_t* srow = &src[(size_t)y * sw * 3];
        float* trow = &tmp[(size_t)y * dw * 3];
        for (int x = 0; x < dw; ++x) {
            const AxisWeights& aw = wx[(size_t)x];
            float r = 0, g = 0, b = 0;
            for (int k = 0; k < (int)aw.w.size(); ++k) {
                const uint8_t* p = srow + (size_t)(aw.start + k) * 3;
                const float wv = aw.w[(size_t)k];
                r += (float)p[0] * wv;
                g += (float)p[1] * wv;
                b += (float)p[2] * wv;
            }
            trow[x * 3 + 0] = r;
            trow[x * 3 + 1] = g;
            trow[x * 3 + 2] = b;
        }
    }
    dst.assign((size_t)dw * dh * 3, 0);
    for (int y = 0; y < dh; ++y) {
        const AxisWeights& aw = wy[(size_t)y];
        uint8_t* drow = &dst[(size_t)y * dw * 3];
        for (int x = 0; x < dw; ++x) {
            float r = 0, g = 0, b = 0;
            for (int k = 0; k < (int)aw.w.size(); ++k) {
                const float* p = &tmp[((size_t)(aw.start + k) * dw + (size_t)x) * 3];
                const float wv = aw.w[(size_t)k];
                r += p[0] * wv;
                g += p[1] * wv;
                b += p[2] * wv;
            }
            auto q = [](float v) {
                int iv = (int)std::lround(v);
                return (uint8_t)std::min(255, std::max(0, iv));
            };
            drow[x * 3 + 0] = q(r);
            drow[x * 3 + 1] = q(g);
            drow[x * 3 + 2] = q(b);
        }
    }
    (void)err;
    return true;
}

}  // namespace

// ---------------------------------------------------------------- preprocessing

bool preprocess_image(const std::string& path, int min_pixels, int max_pixels,
                      std::vector<uint16_t>& patches, ImageInfo& info, std::string& err,
                      std::vector<uint8_t>* resized_rgb) {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    if (!decode_rgb(path, w, h, rgb, err)) return false;
    info = ImageInfo{};
    info.src_w = w;
    info.src_h = h;

    int rh = 0, rw = 0;
    smart_resize(h, w, 32, min_pixels, max_pixels, rh, rw);
    info.resized_h = rh;
    info.resized_w = rw;
    info.grid_h = rh / 16;
    info.grid_w = rw / 16;
    info.n_patches = info.grid_h * info.grid_w;
    info.n_tokens = info.n_patches / 4;

    std::vector<uint8_t> resized;
    if (!resize_rgb(rgb, w, h, rw, rh, resized, err)) return false;
    if (resized_rgb) *resized_rgb = resized;

    // Patchify in merge-block order. A still image duplicates the temporal
    // frame, so the 1536-wide vector is (channel, temporal, py, px).
    const int bpr = info.grid_w / 2;
    const int patch_dim = 1536;
    patches.assign((size_t)info.n_patches * patch_dim, 0);
    const int ps = 16;
    for (int p = 0; p < info.n_patches; ++p) {
        const int blk = p >> 2;
        const int bh = blk / bpr, bw = blk % bpr;
        const int dy = (p >> 1) & 1, dx = p & 1;
        const int y0 = (bh * 2 + dy) * ps;
        const int x0 = (bw * 2 + dx) * ps;
        uint16_t* out = &patches[(size_t)p * patch_dim];
        for (int c = 0; c < 3; ++c) {
            for (int t = 0; t < 2; ++t) {
                for (int py = 0; py < ps; ++py) {
                    const uint8_t* row = &resized[(size_t)(y0 + py) * rw * 3 + (size_t)x0 * 3];
                    for (int px = 0; px < ps; ++px) {
                        const float v = (float)row[px * 3 + c] / 127.5f - 1.0f;
                        const size_t idx = ((size_t)((c * 2 + t) * ps + py) * ps) + (size_t)px;
                        out[idx] = float_to_bf16(v);
                    }
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------- GPU buffers

static MTL::Buffer* grow(Metal& m, MTL::Buffer*& b, size_t bytes) {
    bytes = (bytes + 255) & ~(size_t)255;
    if (b && b->length() >= bytes) return b;
    if (b) b->release();
    b = m.new_buffer(bytes);
    return b;
}

void Vision::ensure(int n_patches) {
    if (n_patches <= cap_patches_) return;
    const size_t N = (size_t)n_patches;
    const size_t pseq = (size_t)((n_patches + 63) / 64) * 64;  // padded key count
    grow(m_, patches_, N * 1536 * 2);
    grow(m_, x_, N * 1024 * 4);
    grow(m_, h_, N * 1024 * 2);
    grow(m_, qf_, N * 1024 * 4);
    grow(m_, kf_, N * 512 * 4);
    grow(m_, vf_, N * 512 * 4);
    grow(m_, qn_, N * 1024 * 2);
    grow(m_, kn_, N * 512 * 2);
    grow(m_, qh_, N * 1024 * 2);
    grow(m_, kh_, pseq * 512 * 2);
    grow(m_, vh_, pseq * 512 * 2);
    grow(m_, ktile_, pseq * 512 * 2);
    grow(m_, attn_h_, N * 1024 * 4);
    grow(m_, attn_b_, N * 1024 * 2);
    grow(m_, proj_, N * 1024 * 4);
    grow(m_, bcx_, N * 3072 * 4);
    grow(m_, gated_, N * 1024 * 4);
    grow(m_, conv_, N * 1024 * 4);
    grow(m_, conv_b_, N * 1024 * 2);
    grow(m_, fg_, N * 2560 * 4);
    grow(m_, fu_, N * 2560 * 4);
    grow(m_, fact_, N * 2560 * 2);
    grow(m_, down_, N * 1024 * 4);
    grow(m_, scores_, pseq * 1024 * 4);
    grow(m_, probs_, pseq * 1024 * 2);
    grow(m_, ln_, N * 1024 * 2);
    grow(m_, lin1_, N * 4096 * 4);
    grow(m_, gelu_, N * 4096 * 2);
    grow(m_, lin2_, N * 2048 * 4);
    grow(m_, feat_, N * 2048 * 2);
    cap_patches_ = n_patches;
}

Vision::~Vision() {}

bool Vision::init(std::string& err) {
    try {
        static const char* names[] = {
            "vision_tower.patchifier.weight",
            "vision_tower.embedding_norm.weight",
            "multi_modal_projector.layer_norm.weight",
            "multi_modal_projector.layer_norm.bias",
            "multi_modal_projector.linear_1.weight",
            "multi_modal_projector.linear_1.bias",
            "multi_modal_projector.linear_2.weight",
            "multi_modal_projector.linear_2.bias",
        };
        for (const char* n : names) w_.get(n);
        for (int l = 0; l < cfg_.layers; ++l) {
            const std::string base = "vision_tower.layers." + std::to_string(l) + ".";
            w_.get(base + (kVisionKind[l] ? "self_attn.q_proj.weight" : "conv.in_proj.weight"));
            w_.get(base + (kVisionKind[l] ? "self_attn.q_layernorm.weight"
                                          : "conv.spatial_conv.weight"));
            w_.get(base + "operator_norm.weight");
            w_.get(base + "ffn_norm.weight");
            w_.get(base + "feed_forward.w1.weight");
            w_.get(base + "feed_forward.w2.weight");
            w_.get(base + "feed_forward.w3.weight");
        }

        // ---- bf16 tile-major cache for the dense GEMMs (tensor cores) -------
        // Every dense weight is copied once into the [N/64][K][64] layout that
        // gemm_mpp consumes, so the tower runs on matmul2d instead of the SIMT
        // gemm_naive. Weights stay in the safetensors mapping too (the CPU
        // reference and the non-GEMM kernels still read them from there).
        std::vector<std::string> tiled;
        tiled.push_back("vision_tower.patchifier.weight");
        for (int l = 0; l < cfg_.layers; ++l) {
            const std::string base = "vision_tower.layers." + std::to_string(l) + ".";
            if (kVisionKind[l]) {
                tiled.push_back(base + "self_attn.q_proj.weight");
                tiled.push_back(base + "self_attn.k_proj.weight");
                tiled.push_back(base + "self_attn.v_proj.weight");
                tiled.push_back(base + "self_attn.out_proj.weight");
            } else {
                tiled.push_back(base + "conv.in_proj.weight");
                tiled.push_back(base + "conv.out_proj.weight");
            }
            tiled.push_back(base + "feed_forward.w1.weight");
            tiled.push_back(base + "feed_forward.w3.weight");
            tiled.push_back(base + "feed_forward.w2.weight");
        }
        tiled.push_back("multi_modal_projector.linear_1.weight");
        tiled.push_back("multi_modal_projector.linear_2.weight");

        uint32_t used = 0;
        for (const std::string& n : tiled) {
            const Tensor& t = w_.get(n);
            if (t.shape.size() != 2 || t.dtype != "BF16")
                throw std::runtime_error("vision: unexpected layout for " + n);
            const uint32_t N = (uint32_t)t.rows(), K = (uint32_t)t.cols();
            if (N % 64u != 0u) throw std::runtime_error("vision: N is not a multiple of 64: " + n);
            used = (used + 255u) & ~255u;
            woff_[n] = used;
            used += N * K * 2u;
        }
        wtile_ = m_.new_buffer(used);
        wtile_used_ = used;
        MTL::CommandBuffer* tcb = m_.begin();
        MTL::ComputeCommandEncoder* tenc = tcb->computeCommandEncoder();
        for (const std::string& n : tiled) {
            const Tensor& t = w_.get(n);
            const uint32_t N = (uint32_t)t.rows(), K = (uint32_t)t.cols();
            auto pipe = m_.pso("tile_bf16");
            tenc->setComputePipelineState(pipe);
            tenc->setBuffer(w_.buffer(), t.begin, 0);
            tenc->setBuffer(wtile_, (NS::UInteger)woff_[n], 1);
            tenc->setBytes(&N, sizeof(N), 2);
            tenc->setBytes(&K, sizeof(K), 3);
            tenc->dispatchThreadgroups(MTL::Size::Make((K + 127u) / 128u, N / 64u, 1),
                                       MTL::Size::Make(256, 1, 1));
        }
        tenc->endEncoding();
        m_.submit(tcb, true);
        printf("vision: bf16 tile-major cache %.1f MB (%zu tensors)\n", used / 1e6, tiled.size());
        // Pre-allocate the activation buffers for a typical image so the first
        // encode does not pay the first-touch cost (grows on demand beyond this).
        ensure(1024);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// ---------------------------------------------------------------- GPU encode

bool Vision::encode(const std::vector<uint16_t>& patches, const ImageInfo& info,
                    std::vector<uint16_t>& features, std::string& err) {
    const int N = info.n_patches;
    const uint32_t H = (uint32_t)cfg_.hidden;
    if (N <= 0 || N % 4 != 0) { err = "vision: patch count must be a positive multiple of 4"; return false; }
    if (N > 16384) {
        err = "vision: " + std::to_string(N) + " patches is too many (max 16384); use --max-pixels";
        return false;
    }
    if (patches.size() != (size_t)N * (size_t)cfg_.patch_dim) { err = "vision: bad patch buffer"; return false; }
    if (info.grid_w < 2 || info.grid_h < 2) { err = "vision: grid too small"; return false; }
    const int M = N / 4;

    ensure(N);
    memcpy(patches_->contents(), patches.data(), (size_t)N * (size_t)cfg_.patch_dim * 2);
    // k/v are kept in head-major [8][pseq][64] views and the MPP attention runs
    // over `pseq` keys (a multiple of 64); rows [N, pseq) must stay zero.
    const uint32_t pseq = (uint32_t)(((N + 63) / 64) * 64);
    memset(kh_->contents(), 0, (size_t)pseq * 512 * 2);
    memset(vh_->contents(), 0, (size_t)pseq * 512 * 2);
    constexpr uint32_t kQBlock = 1024;

    const uint32_t gw = (uint32_t)info.grid_w, gh = (uint32_t)info.grid_h;
    MTL::CommandBuffer* cb = m_.begin();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    const bool dbg = getenv("MIORI_VISION_DEBUG") != nullptr;
    const bool prof = getenv("MIORI_VISION_PROFILE") != nullptr;
    auto tprev = std::chrono::steady_clock::now();
    auto mark = [&](const char* tag) {
        if (!prof) return;
        enc->endEncoding();
        m_.submit(cb, true);
        const auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "[prof] %-22s %7.2f ms\n", tag,
                std::chrono::duration<double>(now - tprev).count() * 1000.0);
        tprev = now;
        cb = m_.begin();
        enc = cb->computeCommandEncoder();
    };
    auto dumpf = [&](const char* tag, MTL::Buffer* b, size_t n) {
        if (!dbg) return;
        enc->endEncoding();
        m_.submit(cb, true);
        const float* p = (const float*)b->contents();
        double ss = 0.0; float mn = 1e30f, mx = -1e30f; int bad = 0;
        for (size_t i = 0; i < n; ++i) {
            const float v = p[i];
            if (!std::isfinite(v)) ++bad;
            else { ss += (double)v * v; mn = std::min(mn, v); mx = std::max(mx, v); }
        }
        fprintf(stderr, "[gpu] %-18s n=%zu rms=%.4f min=%.4f max=%.4f bad=%d\n", tag, n,
                std::sqrt(ss / (double)n), mn, mx, bad);
        cb = m_.begin();
        enc = cb->computeCommandEncoder();
    };

    (void)0;
    auto wname = [&](int l, const char* suffix) {
        return "vision_tower.layers." + std::to_string(l) + "." + suffix;
    };
    // Dense matmuls go through the tensor-core path whenever the operands are
    // contiguous in the layout gemm_mpp assumes (A [M][K], C [M][N], N % 64 == 0).
    // Attention QK/PV keep gemm_naive: they use per-head column/byte offsets and
    // a [64][seq] "weight" layout that MPP cannot express without a repack.
    auto gemm = [&](MTL::Buffer* A, uint32_t lda, const std::string& name, MTL::Buffer* C,
                    uint32_t ldc, uint32_t m, uint32_t n, uint32_t k) {
        auto it = woff_.find(name);
        if (it != woff_.end() && lda == k && ldc == n && (n % 64u) == 0u && m > 0) {
            gemm_mpp(m_, enc, A, wtile_, C, m, n, k, it->second);
            return;
        }
        const Tensor& t = w_.get(name);
        gemm_naive(m_, enc, A, w_.buffer(), C, m, n, k, lda, k, ldc, 0, (uint32_t)t.begin, 0);
    };

    // patchifier + embedding_norm
    gemm(patches_, 1536, "vision_tower.patchifier.weight", x_, 1024, (uint32_t)N, H, 1536);
    dumpf("patchifier", x_, (size_t)N * H);
    rmsnorm(m_, enc, x_, w_, w_.get("vision_tower.embedding_norm.weight"), h_,
            (uint32_t)N, H, cfg_.norm_eps);
    // embedding_norm output *is* the initial residual stream
    to_f32(m_, enc, h_, x_, (uint32_t)N * H);
    mark("patchifier+norm");

    for (int l = 0; l < cfg_.layers; ++l) {
        if (!kVisionKind[l]) {
            // gated 3x3 depthwise conv in raster order, back to merge-block order
            rmsnorm(m_, enc, x_, w_, w_.get(wname(l, "operator_norm.weight")), h_,
                    (uint32_t)N, H, cfg_.norm_eps);
            gemm(h_, 1024, wname(l, "conv.in_proj.weight"), bcx_, 3072, (uint32_t)N, 3072, H);
            conv_gate(m_, enc, bcx_, gated_, (uint32_t)N, H);
            {
                auto p = m_.pso("vision_conv3x3");
                enc->setComputePipelineState(p);
                enc->setBuffer(gated_, 0, 0);
                enc->setBuffer(bcx_, 0, 1);
                const Tensor& kt = w_.get(wname(l, "conv.spatial_conv.weight"));
                enc->setBuffer(w_.buffer(), kt.begin, 2);
                enc->setBuffer(conv_, 0, 3);
                uint32_t n = (uint32_t)N;
                enc->setBytes(&n, sizeof(n), 4);
                enc->setBytes(&gw, sizeof(gw), 5);
                enc->setBytes(&gh, sizeof(gh), 6);
                enc->setBytes(&H, sizeof(H), 7);
                enc->dispatchThreads(MTL::Size::Make((size_t)N * H, 1, 1),
                                     MTL::Size::Make(256, 1, 1));
            }
            to_bf16(m_, enc, conv_, conv_b_, (uint32_t)N * H);
            gemm(conv_b_, 1024, wname(l, "conv.out_proj.weight"), proj_, 1024, (uint32_t)N, H, H);
            add(m_, enc, x_, proj_, x_, (uint32_t)N * H);
            if (l < 2) { char tg[32]; snprintf(tg, sizeof(tg), "layer %d op", l); dumpf(tg, x_, (size_t)N * H); }
            mark("conv op");
        } else {
            rmsnorm(m_, enc, x_, w_, w_.get(wname(l, "operator_norm.weight")), h_,
                    (uint32_t)N, H, cfg_.norm_eps);
            gemm(h_, 1024, wname(l, "self_attn.q_proj.weight"), qf_, 1024, (uint32_t)N,
                 (uint32_t)cfg_.heads * 64u, H);
            gemm(h_, 1024, wname(l, "self_attn.k_proj.weight"), kf_, 512, (uint32_t)N,
                 (uint32_t)cfg_.kv_heads * 64u, H);
            gemm(h_, 1024, wname(l, "self_attn.v_proj.weight"), vf_, 512, (uint32_t)N,
                 (uint32_t)cfg_.kv_heads * 64u, H);
            // per-head RMSNorm on q and k (rows = patches * heads, width 64)
            rmsnorm(m_, enc, qf_, w_, w_.get(wname(l, "self_attn.q_layernorm.weight")), qn_,
                    (uint32_t)N * (uint32_t)cfg_.heads, 64, cfg_.norm_eps);
            rmsnorm(m_, enc, kf_, w_, w_.get(wname(l, "self_attn.k_layernorm.weight")), kn_,
                    (uint32_t)N * (uint32_t)cfg_.kv_heads, 64, cfg_.norm_eps);
            {
                float theta = cfg_.rope_theta;
                for (uint32_t heads : {(uint32_t)cfg_.heads, (uint32_t)cfg_.kv_heads}) {
                    MTL::Buffer* dst = heads == (uint32_t)cfg_.heads ? qn_ : kn_;
                    auto p = m_.pso("vision_rope");
                    enc->setComputePipelineState(p);
                    enc->setBuffer(dst, 0, 0);
                    uint32_t n = (uint32_t)N;
                    uint32_t dim = 64;
                    enc->setBytes(&n, sizeof(n), 1);
                    enc->setBytes(&heads, sizeof(heads), 2);
                    enc->setBytes(&dim, sizeof(dim), 3);
                    enc->setBytes(&gw, sizeof(gw), 4);
                    enc->setBytes(&theta, sizeof(theta), 5);
                    enc->dispatchThreads(MTL::Size::Make((size_t)N * heads * 32u, 1, 1),
                                         MTL::Size::Make(256, 1, 1));
                }
            }
            // --- head-major views: one contiguous [rows][64] block per head ----
            {
                auto p = m_.pso("vision_to_heads");
                enc->setComputePipelineState(p);
                enc->setBuffer(qn_, 0, 0);
                enc->setBuffer(qh_, 0, 1);
                uint32_t n = (uint32_t)N, heads = (uint32_t)cfg_.heads, dstride = (uint32_t)N;
                enc->setBytes(&n, sizeof(n), 2);
                enc->setBytes(&heads, sizeof(heads), 3);
                enc->setBytes(&dstride, sizeof(dstride), 4);
                enc->dispatchThreads(MTL::Size::Make((size_t)N * 1024, 1, 1),
                                     MTL::Size::Make(256, 1, 1));
            }
            {
                auto p = m_.pso("vision_to_heads");
                enc->setComputePipelineState(p);
                enc->setBuffer(kn_, 0, 0);
                enc->setBuffer(kh_, 0, 1);
                uint32_t n = (uint32_t)N, heads = (uint32_t)cfg_.kv_heads, dstride = pseq;
                enc->setBytes(&n, sizeof(n), 2);
                enc->setBytes(&heads, sizeof(heads), 3);
                enc->setBytes(&dstride, sizeof(dstride), 4);
                enc->dispatchThreads(MTL::Size::Make((size_t)N * 512, 1, 1),
                                     MTL::Size::Make(256, 1, 1));
            }
            {
                auto p = m_.pso("vision_v_heads");
                enc->setComputePipelineState(p);
                enc->setBuffer(vf_, 0, 0);
                enc->setBuffer(vh_, 0, 1);
                uint32_t n = (uint32_t)N, dstride = pseq;
                enc->setBytes(&n, sizeof(n), 2);
                enc->setBytes(&dstride, sizeof(dstride), 3);
                enc->dispatchThreads(MTL::Size::Make((size_t)N * 512, 1, 1),
                                     MTL::Size::Make(256, 1, 1));
            }
            // tile each kv head into [pseq/64][64][64] for the QK matmul
            for (uint32_t hkv = 0; hkv < (uint32_t)cfg_.kv_heads; ++hkv) {
                auto p = m_.pso("tile_bf16");
                enc->setComputePipelineState(p);
                enc->setBuffer(kh_, (NS::UInteger)((size_t)hkv * pseq * 64 * 2), 0);
                enc->setBuffer(ktile_,
                               (NS::UInteger)((size_t)hkv * (pseq / 64) * 64 * 64 * 2), 1);
                uint32_t nn = pseq, kk = 64;
                enc->setBytes(&nn, sizeof(nn), 2);
                enc->setBytes(&kk, sizeof(kk), 3);
                enc->dispatchThreadgroups(MTL::Size::Make(1, pseq / 64, 1),
                                          MTL::Size::Make(256, 1, 1));
            }
            // bidirectional GQA on tensor cores; keys are padded to a multiple of 64
            for (int q0 = 0; q0 < N; q0 += (int)kQBlock) {
                const uint32_t mq = (uint32_t)std::min<int>(kQBlock, N - q0);
                for (int hh = 0; hh < cfg_.heads; ++hh) {
                    const uint32_t kvh = (uint32_t)(hh / (cfg_.heads / cfg_.kv_heads));
                    gemm_mpp_off(m_, enc, qh_, ktile_, scores_, mq, pseq, 64,
                                 (uint32_t)(((size_t)hh * N + q0) * 64 * 2),
                                 (uint32_t)((size_t)kvh * (pseq / 64) * 64 * 64 * 2), 0);
                    {
                        auto p = m_.pso("vision_softmax");
                        enc->setComputePipelineState(p);
                        enc->setBuffer(scores_, 0, 0);
                        enc->setBuffer(probs_, 0, 1);
                        float scale = 1.0f / std::sqrt(64.0f);
                        uint32_t limit = (uint32_t)N;
                        enc->setBytes(&pseq, sizeof(pseq), 2);
                        enc->setBytes(&scale, sizeof(scale), 3);
                        enc->setBytes(&limit, sizeof(limit), 4);
                        enc->dispatchThreadgroups(MTL::Size::Make(mq, 1, 1),
                                                  MTL::Size::Make(256, 1, 1));
                    }
                    // MPP writes f32 into the head-major attention buffer
                    gemm_mpp_off(m_, enc, probs_, vh_, attn_h_, mq, 64, pseq, 0,
                                 (uint32_t)((size_t)kvh * pseq * 64 * 2),
                                 (uint32_t)(((size_t)hh * N + q0) * 64 * 4));
                }
            }
            {
                auto p = m_.pso("vision_from_heads");
                enc->setComputePipelineState(p);
                enc->setBuffer(attn_h_, 0, 0);
                enc->setBuffer(attn_b_, 0, 1);
                uint32_t n = (uint32_t)N, heads = (uint32_t)cfg_.heads;
                enc->setBytes(&n, sizeof(n), 2);
                enc->setBytes(&heads, sizeof(heads), 3);
                enc->dispatchThreads(MTL::Size::Make((size_t)N * 1024, 1, 1),
                                     MTL::Size::Make(256, 1, 1));
            }
            gemm(attn_b_, 1024, wname(l, "self_attn.out_proj.weight"), proj_, 1024,
                 (uint32_t)N, H, H);
            add(m_, enc, x_, proj_, x_, (uint32_t)N * H);
            if (l < 2) { char tg[32]; snprintf(tg, sizeof(tg), "layer %d op", l); dumpf(tg, x_, (size_t)N * H); }
            mark("attn op");
        }
        // pre-norm only feed-forward
        rmsnorm(m_, enc, x_, w_, w_.get(wname(l, "ffn_norm.weight")), h_,
                (uint32_t)N, H, cfg_.norm_eps);
        gemm(h_, 1024, wname(l, "feed_forward.w1.weight"), fg_, 2560, (uint32_t)N, 2560, H);
        gemm(h_, 1024, wname(l, "feed_forward.w3.weight"), fu_, 2560, (uint32_t)N, 2560, H);
        silu_mul(m_, enc, fg_, fu_, fact_, (uint32_t)N * 2560);
        gemm(fact_, 2560, wname(l, "feed_forward.w2.weight"), down_, 1024, (uint32_t)N, H, 2560);
        add(m_, enc, x_, down_, x_, (uint32_t)N * H);
        if (l < 2 || l == 13) { char tg[32]; snprintf(tg, sizeof(tg), "layer %d ffn", l); dumpf(tg, x_, (size_t)N * H); }
        mark("ffn");
    }

    // projector: LayerNorm -> reshape 2x2 blocks to 4096 -> linear_1 -> GELU -> linear_2
    {
        // the tower's output is bf16 in the reference; convert before LayerNorm
        to_bf16(m_, enc, x_, conv_b_, (uint32_t)N * H);
        auto p = m_.pso("vision_layernorm");
        enc->setComputePipelineState(p);
        enc->setBuffer(conv_b_, 0, 0);
        const Tensor& lnw = w_.get("multi_modal_projector.layer_norm.weight");
        const Tensor& lnb = w_.get("multi_modal_projector.layer_norm.bias");
        enc->setBuffer(w_.buffer(), lnw.begin, 1);
        enc->setBuffer(w_.buffer(), lnb.begin, 2);
        enc->setBuffer(ln_, 0, 3);
        enc->setBytes(&H, sizeof(H), 4);
        float eps = cfg_.proj_eps;
        enc->setBytes(&eps, sizeof(eps), 5);
        enc->dispatchThreadgroups(MTL::Size::Make((size_t)N, 1, 1), MTL::Size::Make(256, 1, 1));
    }
    // ln_ is [N][1024] contiguous, i.e. already [N/4][4096] in block order.
    dumpf("proj ln(in)", x_, (size_t)N * H);
    gemm(ln_, 4096, "multi_modal_projector.linear_1.weight", lin1_, 4096, (uint32_t)M, 4096, 4096);
    dumpf("linear_1", lin1_, (size_t)M * 4096);
    {
        auto p = m_.pso("vision_bias_gelu");
        enc->setComputePipelineState(p);
        enc->setBuffer(lin1_, 0, 0);
        const Tensor& b = w_.get("multi_modal_projector.linear_1.bias");
        enc->setBuffer(w_.buffer(), b.begin, 1);
        enc->setBuffer(gelu_, 0, 2);
        uint32_t total = (uint32_t)M * 4096u, cols = 4096;
        enc->setBytes(&total, sizeof(total), 3);
        enc->setBytes(&cols, sizeof(cols), 4);
        enc->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
    }
    gemm(gelu_, 4096, "multi_modal_projector.linear_2.weight", lin2_, 2048, (uint32_t)M, 2048, 4096);
    dumpf("linear_2", lin2_, (size_t)M * 2048);
    {
        auto p = m_.pso("vision_bias");
        enc->setComputePipelineState(p);
        enc->setBuffer(lin2_, 0, 0);
        const Tensor& b = w_.get("multi_modal_projector.linear_2.bias");
        enc->setBuffer(w_.buffer(), b.begin, 1);
        enc->setBuffer(feat_, 0, 2);
        uint32_t total = (uint32_t)M * 2048u, cols = 2048;
        enc->setBytes(&total, sizeof(total), 3);
        enc->setBytes(&cols, sizeof(cols), 4);
        enc->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
    }

    mark("projector");
    enc->endEncoding();
    m_.submit(cb, true);

    features.resize((size_t)M * 2048);
    memcpy(features.data(), feat_->contents(), features.size() * 2);
    return true;
}

// ---------------------------------------------------------------- CPU reference
//
// Deliberately written from the reference module semantics (not from the Metal
// code): same weights, fp32 throughout. vision-test compares the two.

namespace {

using FVec = std::vector<float>;

FVec wf32(const Weights& w, const std::string& name) {
    const Tensor& t = w.get(name);
    const size_t n = t.bytes() / 2;
    FVec out(n);
    const uint16_t* p = (const uint16_t*)t.ptr;
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_float(p[i]);
    return out;
}

void mm(const FVec& A, const FVec& B, FVec& C, int M, int N, int K) {
    C.assign((size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        const float* a = &A[(size_t)m * K];
        float* c = &C[(size_t)m * N];
        for (int n = 0; n < N; ++n) {
            const float* b = &B[(size_t)n * K];
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += a[k] * b[k];
            c[n] = acc;
        }
    }
}

void rmsnorm_inplace(FVec& x, const FVec& g, int rows, int n, float eps) {
    for (int r = 0; r < rows; ++r) {
        float* row = &x[(size_t)r * n];
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double)row[i] * row[i];
        const float inv = (float)(1.0 / std::sqrt(ss / n + eps));
        for (int i = 0; i < n; ++i) row[i] = row[i] * inv * g[(size_t)i];
    }
}

void layernorm(const FVec& in, const FVec& g, const FVec& b, FVec& out, int rows, int n,
               float eps) {
    out.assign(in.size(), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* x = &in[(size_t)r * n];
        double s = 0.0, sq = 0.0;
        for (int i = 0; i < n; ++i) { s += x[i]; sq += (double)x[i] * x[i]; }
        const double mean = s / n;
        const double var = sq / n - mean * mean;
        const float inv = (float)(1.0 / std::sqrt(var + eps));
        float* y = &out[(size_t)r * n];
        for (int i = 0; i < n; ++i) y[i] = (float)((x[i] - mean) * inv) * g[(size_t)i] + b[(size_t)i];
    }
}

void rope_axial(FVec& x, int N, int heads, int gw, float theta) {
    const int bpr = gw / 2;
    for (int p = 0; p < N; ++p) {
        const int blk = p >> 2;
        const int bh = blk / bpr, bw = blk % bpr;
        const float h = (float)(bh * 2 + ((p >> 1) & 1));
        const float w = (float)(bw * 2 + (p & 1));
        for (int hd = 0; hd < heads; ++hd) {
            float* head = &x[((size_t)p * heads + hd) * 64];
            for (int d = 0; d < 32; ++d) {
                const int i = d % 16;
                const float inv = std::pow(theta, -(float)i / 16.0f);
                const float ang = ((d % 32) < 16 ? h : w) * inv;
                const float c = std::cos(ang), s = std::sin(ang);
                const float x0 = head[d], x1 = head[d + 32];
                head[d] = x0 * c - x1 * s;
                head[d + 32] = x1 * c + x0 * s;
            }
        }
    }
}

}  // namespace

bool Vision::cpu_reference(const std::vector<uint16_t>& patches, const ImageInfo& info,
                           std::vector<float>& features, std::string& err) {
    const int N = info.n_patches;
    const int H = cfg_.hidden;
    if (N <= 0 || N % 4 != 0) { err = "cpu_reference: bad patch count"; return false; }
    const int M = N / 4;
    const int gw = info.grid_w, gh = info.grid_h;
    const float eps = cfg_.norm_eps;

    FVec patch((size_t)N * 1536);
    for (size_t i = 0; i < patch.size(); ++i) patch[i] = bf16_to_float(patches[i]);

    FVec x, tmp;
    const bool verbose = getenv("MIORI_VISION_DEBUG") != nullptr;
    auto dbg = [&](const char* tag) {
        if (!verbose) return;
        double ss = 0.0; float mn = 1e30f, mx = -1e30f; bool bad = false;
        for (float v : x) { if (!std::isfinite(v)) bad = true; ss += (double)v*v; mn = std::min(mn, v); mx = std::max(mx, v); }
        fprintf(stderr, "[cpu] %-16s l2=%.4f min=%.4f max=%.4f %s\n", tag, std::sqrt(ss), mn, mx, bad ? "NONFINITE" : "");
    };
    {
        FVec W = wf32(w_, "vision_tower.patchifier.weight");
        mm(patch, W, x, N, H, 1536);
    }
    dbg("patchifier");
    {
        FVec g = wf32(w_, "vision_tower.embedding_norm.weight");
        rmsnorm_inplace(x, g, N, H, eps);
    }
    dbg("embed_norm");

    for (int l = 0; l < cfg_.layers; ++l) {
        const std::string base = "vision_tower.layers." + std::to_string(l) + ".";
        FVec normed = x;
        rmsnorm_inplace(normed, wf32(w_, base + "operator_norm.weight"), N, H, eps);
        FVec proj;
        if (!kVisionKind[l]) {
            FVec W = wf32(w_, base + "conv.in_proj.weight");
            FVec bcx;
            mm(normed, W, bcx, N, 3 * H, H);
            FVec gated((size_t)N * H);
            for (int p = 0; p < N; ++p)
                for (int c = 0; c < H; ++c)
                    gated[(size_t)p * H + c] = bcx[(size_t)p * 3 * H + c] *
                                               bcx[(size_t)p * 3 * H + 2 * H + c];
            // merge-block -> raster
            FVec raster((size_t)gh * gw * H, 0.0f);
            const int bpr = gw / 2;
            for (int p = 0; p < N; ++p) {
                const int blk = p >> 2;
                const int bh = blk / bpr, bw = blk % bpr;
                const int h = bh * 2 + ((p >> 1) & 1);
                const int w = bw * 2 + (p & 1);
                memcpy(&raster[((size_t)h * gw + w) * H], &gated[(size_t)p * H],
                       (size_t)H * sizeof(float));
            }
            FVec conv((size_t)gh * gw * H, 0.0f);
            FVec Wc = wf32(w_, base + "conv.spatial_conv.weight");
            for (int h = 0; h < gh; ++h) {
                for (int w = 0; w < gw; ++w) {
                    for (int c = 0; c < H; ++c) {
                        float acc = 0.0f;
                        for (int kh = -1; kh <= 1; ++kh) {
                            const int nh = h + kh;
                            if (nh < 0 || nh >= gh) continue;
                            for (int kw = -1; kw <= 1; ++kw) {
                                const int nw = w + kw;
                                if (nw < 0 || nw >= gw) continue;
                                acc += raster[((size_t)nh * gw + nw) * H + c] *
                                       Wc[(size_t)c * 9 + (size_t)(kh + 1) * 3 + (size_t)(kw + 1)];
                            }
                        }
                        conv[((size_t)h * gw + w) * H + c] = acc;
                    }
                }
            }
            FVec outc((size_t)N * H);
            for (int p = 0; p < N; ++p) {
                const int blk = p >> 2;
                const int bh = blk / bpr, bw = blk % bpr;
                const int h = bh * 2 + ((p >> 1) & 1);
                const int w = bw * 2 + (p & 1);
                for (int c = 0; c < H; ++c)
                    outc[(size_t)p * H + c] =
                        bcx[(size_t)p * 3 * H + H + c] * conv[((size_t)h * gw + w) * H + c];
            }
            FVec Wo = wf32(w_, base + "conv.out_proj.weight");
            mm(outc, Wo, proj, N, H, H);
        } else {
            FVec Wq = wf32(w_, base + "self_attn.q_proj.weight");
            FVec Wk = wf32(w_, base + "self_attn.k_proj.weight");
            FVec Wv = wf32(w_, base + "self_attn.v_proj.weight");
            FVec q, k, v;
            mm(normed, Wq, q, N, cfg_.heads * 64, H);
            mm(normed, Wk, k, N, cfg_.kv_heads * 64, H);
            mm(normed, Wv, v, N, cfg_.kv_heads * 64, H);
            rmsnorm_inplace(q, wf32(w_, base + "self_attn.q_layernorm.weight"), N * cfg_.heads,
                            64, eps);
            rmsnorm_inplace(k, wf32(w_, base + "self_attn.k_layernorm.weight"), N * cfg_.kv_heads,
                            64, eps);
            rope_axial(q, N, cfg_.heads, gw, cfg_.rope_theta);
            rope_axial(k, N, cfg_.kv_heads, gw, cfg_.rope_theta);
            FVec out((size_t)N * H, 0.0f);
            const float scale = 1.0f / std::sqrt(64.0f);
            std::vector<float> sc((size_t)N);
            for (int hd = 0; hd < cfg_.heads; ++hd) {
                const int kvh = hd / (cfg_.heads / cfg_.kv_heads);
                for (int i = 0; i < N; ++i) {
                    float mx = -INFINITY;
                    for (int j = 0; j < N; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < 64; ++d)
                            s += q[((size_t)i * cfg_.heads + hd) * 64 + d] *
                                 k[((size_t)j * cfg_.kv_heads + kvh) * 64 + d];
                        s *= scale;
                        sc[(size_t)j] = s;
                        mx = std::max(mx, s);
                    }
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) { sc[(size_t)j] = std::exp(sc[(size_t)j] - mx); sum += sc[(size_t)j]; }
                    for (int j = 0; j < N; ++j) sc[(size_t)j] /= sum;
                    for (int d = 0; d < 64; ++d) {
                        float a = 0.0f;
                        for (int j = 0; j < N; ++j)
                            a += sc[(size_t)j] * v[((size_t)j * cfg_.kv_heads + kvh) * 64 + d];
                        out[(size_t)i * H + hd * 64 + d] = a;
                    }
                }
            }
            FVec Wo = wf32(w_, base + "self_attn.out_proj.weight");
            mm(out, Wo, proj, N, H, H);
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
        if (l < 3 || l == 13) { char tag[32]; snprintf(tag, sizeof(tag), "layer %d op", l); dbg(tag); }

        FVec n2 = x;
        rmsnorm_inplace(n2, wf32(w_, base + "ffn_norm.weight"), N, H, eps);
        FVec g, u, act;
        mm(n2, wf32(w_, base + "feed_forward.w1.weight"), g, N, cfg_.inter, H);
        mm(n2, wf32(w_, base + "feed_forward.w3.weight"), u, N, cfg_.inter, H);
        act.resize(g.size());
        for (size_t i = 0; i < g.size(); ++i)
            act[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
        FVec d;
        mm(act, wf32(w_, base + "feed_forward.w2.weight"), d, N, H, cfg_.inter);
        for (size_t i = 0; i < x.size(); ++i) x[i] += d[i];
        if (l < 3 || l == 13) { char tag[32]; snprintf(tag, sizeof(tag), "layer %d ffn", l); dbg(tag); }
    }

    auto dbgv = [&](const char* tag, const FVec& v) {
        if (!verbose) return;
        double ss = 0.0; float mn = 1e30f, mx = -1e30f; int bad = 0;
        for (float z : v) { if (!std::isfinite(z)) ++bad; else { ss += (double)z*z; mn = std::min(mn, z); mx = std::max(mx, z);} }
        fprintf(stderr, "[cpu] %-16s n=%zu rms=%.4f min=%.4f max=%.4f bad=%d\n", tag, v.size(),
                std::sqrt(ss / (double)v.size()), mn, mx, bad);
    };
    dbgv("proj input", x);
    FVec ln;
    layernorm(x, wf32(w_, "multi_modal_projector.layer_norm.weight"),
              wf32(w_, "multi_modal_projector.layer_norm.bias"), ln, N, H, cfg_.proj_eps);
    dbgv("proj ln", ln);
    // ln is [N][1024] == [N/4][4096] in block order
    FVec l1;
    mm(ln, wf32(w_, "multi_modal_projector.linear_1.weight"), l1, M, 4096, 4096);
    dbgv("linear_1", l1);
    {
        FVec b = wf32(w_, "multi_modal_projector.linear_1.bias");
        for (int m = 0; m < M; ++m)
            for (int i = 0; i < 4096; ++i) {
                float& val = l1[(size_t)m * 4096 + i];
                val += b[(size_t)i];
                val = 0.5f * val * (1.0f + std::erf(val * 0.7071067811865476f));
            }
    }
    FVec l2;
    dbgv("gelu", l1);
    mm(l1, wf32(w_, "multi_modal_projector.linear_2.weight"), l2, M, 2048, 4096);
    dbgv("linear_2", l2);
    FVec b2 = wf32(w_, "multi_modal_projector.linear_2.bias");
    features.assign((size_t)M * 2048, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < 2048; ++i)
            features[(size_t)m * 2048 + i] = l2[(size_t)m * 2048 + i] + b2[(size_t)i];
    return true;
}

}  // namespace miori
