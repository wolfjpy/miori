#include "metal.h"
#include "weights.h"
#include "ops.h"
#include "cpu_ref.h"
#include "tokenizer.h"
#include "engine.h"
#include "batch.h"
#include "vision.h"
#include "m6.h"
#include "server.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>

using namespace miori;

static void fill_bf16(MTL::Buffer* b, size_t n, float (*fn)(size_t)) {
    uint16_t* p = (uint16_t*)b->contents();
    for (size_t i = 0; i < n; ++i) p[i] = float_to_bf16(fn(i));
}

static float xfn(size_t k) { return ((float)((k * 37) % 101) - 50.0f) / 50.0f; }

static void test_gemv(Metal& m, Weights& w, bool eight, bool four = false) {
    const Tensor& q = eight ? w.get("language_model.model.embed_tokens.weight")
                            : w.get("language_model.model.layers.2.self_attn.q_proj.weight");
    const Tensor& s = eight ? w.get("language_model.model.embed_tokens.scales")
                            : w.get("language_model.model.layers.2.self_attn.q_proj.scales");
    const Tensor& b = eight ? w.get("language_model.model.embed_tokens.biases")
                            : w.get("language_model.model.layers.2.self_attn.q_proj.biases");
    const uint32_t N = (uint32_t)q.rows();
    const uint32_t K = eight ? 2048u : (uint32_t)(q.cols() * 8u);
    printf("test gemv%s: N=%u K=%u\n", eight ? "8" : "4", N, K);

    MTL::Buffer* x = m.new_buffer((size_t)K * 2);
    MTL::Buffer* y = m.new_buffer((size_t)N * 4);
    fill_bf16(x, K, xfn);

    auto cb = m.begin();
    auto e = cb->computeCommandEncoder();
    if (eight) gemv8(m, e, w, q, s, b, x, y, N, K);
    else if (four) gemv4x4(m, e, w, q, s, b, x, y, N, K);
    else gemv4(m, e, w, q, s, b, x, y, N, K);
    e->endEncoding();
    m.submit(cb, true);

    const float* gy = (const float*)y->contents();
    std::vector<float> gx(K);
    for (uint32_t k = 0; k < K; ++k) gx[k] = bf16_to_float(((uint16_t*)x->contents())[k]);

    double max_abs = 0.0, max_rel = 0.0;
    int checked = 0;
    for (uint32_t n = 0; n < N; n += eight ? 997u : 1u) {
        double ref = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
            float wv = eight ? dequant8_elem(q, s, b, n, k, K)
                             : dequant4_elem(q, s, b, n, k, K);
            ref += (double)wv * gx[k];
        }
        double err = std::fabs(ref - gy[n]);
        max_abs = std::max(max_abs, err);
        max_rel = std::max(max_rel, err / (std::fabs(ref) + 1e-6));
        ++checked;
    }
    printf("checked %d rows: max_abs=%.6g max_rel=%.6g  y[0]=%.6g\n",
           checked, max_abs, max_rel, gy[0]);
}


// ---------------------------------------------------------------- vision ----

static void append_enc(const Tokenizer& t, const std::string& text, std::vector<int>& ids) {
    auto v = t.encode(text);
    ids.insert(ids.end(), v.begin(), v.end());
}

static void print_feature_stats(const std::vector<uint16_t>& feat, int nt) {
    double ss = 0.0, sum = 0.0;
    float mn = 1e30f, mx = -1e30f;
    for (uint16_t v : feat) {
        const float f = bf16_to_float(v);
        ss += (double)f * f;
        sum += f;
        mn = std::min(mn, f);
        mx = std::max(mx, f);
    }
    const double n = (double)feat.size();
    const double mean = sum / n;
    printf("features: %d tokens x 2048, L2=%.3f mean=%.4f std=%.4f min=%.3f max=%.3f\n", nt,
           std::sqrt(ss), mean, std::sqrt(std::max(0.0, ss / n - mean * mean)), mn, mx);
    for (int r = 0; r < std::min(nt, 3); ++r) {
        double rss = 0.0;
        for (int i = 0; i < 2048; ++i) {
            const float f = bf16_to_float(feat[(size_t)r * 2048 + i]);
            rss += (double)f * f;
        }
        printf("  token %d: L2=%.3f  ", r, std::sqrt(rss));
        printf("first8:");
        for (int i = 0; i < 8; ++i) printf(" %.4f", bf16_to_float(feat[(size_t)r * 2048 + i]));
        printf("\n");
    }
}

static void write_ppm(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "vision: cannot write %s\n", path.c_str()); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
    printf("wrote resized RGB (%dx%d) to %s\n", w, h, path.c_str());
}

static bool compare_features(const std::vector<uint16_t>& gpu, const std::vector<float>& cpu,
                             int nt, std::string& err) {
    if (gpu.size() != cpu.size()) { err = "feature size mismatch"; return false; }
    double worst_cos = 2.0, max_abs = 0.0;
    int bad = 0;
    for (int r = 0; r < nt; ++r) {
        double dot = 0.0, ng = 0.0, nc = 0.0;
        for (int i = 0; i < 2048; ++i) {
            const double g = bf16_to_float(gpu[(size_t)r * 2048 + i]);
            const double c = cpu[(size_t)r * 2048 + i];
            if (!std::isfinite(g) || !std::isfinite(c)) { ++bad; continue; }
            dot += g * c;
            ng += g * g;
            nc += c * c;
            max_abs = std::max(max_abs, std::fabs(g - c));
        }
        if (ng <= 0.0 || nc <= 0.0) { ++bad; continue; }
        const double cos = dot / (std::sqrt(ng) * std::sqrt(nc));
        worst_cos = std::min(worst_cos, cos);
    }
    printf("cpu-ref: worst per-token cosine = %.6f, max |gpu-cpu| = %.5f, non-finite=%d\n",
           worst_cos, max_abs, bad);
    return worst_cos > 0.99 && bad == 0;
}

static void tensor_stats(const Weights& w, const char* name) {
    const Tensor* t = w.find(name);
    if (!t) { printf("%-52s MISSING\n", name); return; }
    if (t->dtype != "BF16") { printf("%-52s dtype=%s\n", name, t->dtype.c_str()); return; }
    const uint16_t* p = (const uint16_t*)t->ptr;
    const size_t n = t->bytes() / 2;
    double ss = 0.0, sum = 0.0;
    float mn = 1e30f, mx = -1e30f;
    bool bad = false;
    for (size_t i = 0; i < n; ++i) {
        const float v = bf16_to_float(p[i]);
        if (!std::isfinite(v)) bad = true;
        ss += (double)v * v; sum += v;
        mn = std::min(mn, v); mx = std::max(mx, v);
    }
    printf("%-52s n=%-8zu rms=%.5f mean=%.5f min=%.4f max=%.4f%s\n", name, n,
           std::sqrt(ss / n), sum / n, mn, mx, bad ? "  NONFINITE" : "");
}

static std::vector<uint16_t> synthetic_patches(int grid_w, int grid_h) {
    const int n = grid_w * grid_h;
    std::vector<uint16_t> p((size_t)n * 1536);
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 1536; ++c) {
            const float v = 0.6f * std::sin(0.013f * (c * 7 + i * 29)) +
                            0.3f * std::cos(0.037f * (c * 3 - i * 11)) +
                            0.05f * (float)((c % 19) - 9);
            p[(size_t)i * 1536 + c] = float_to_bf16(v);
        }
    }
    return p;
}

static std::vector<int> build_image_prompt(const Tokenizer& t, const std::string& prompt,
                                           int n_image_tokens) {
    std::vector<int> ids;
    ids.push_back(kBosId);
    ids.push_back(kImStartId);
    append_enc(t, "system\nYou are a helpful assistant.", ids);
    ids.push_back(kImEndId);
    append_enc(t, "\n", ids);
    ids.push_back(kImStartId);
    append_enc(t, "user\n", ids);
    ids.push_back(kVisionStartId);
    for (int i = 0; i < n_image_tokens; ++i) ids.push_back(kImageId);
    ids.push_back(kVisionEndId);
    append_enc(t, prompt, ids);
    ids.push_back(kImEndId);
    append_enc(t, "\n", ids);
    ids.push_back(kImStartId);
    append_enc(t, "assistant\n", ids);
    ids.push_back(kThinkId);
    ids.push_back(kEndThinkId);
    return ids;
}

static int cmd_vision_test(Metal& m, Weights& w, int argc, char** argv) {
    std::string image;
    int min_pixels = 65536, max_pixels = 16777216;
    bool cpu_ref = false, text_check = false;
    std::string dump;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) image = argv[++i];
        else if (a == "--min-pixels" && i + 1 < argc) min_pixels = atoi(argv[++i]);
        else if (a == "--max-pixels" && i + 1 < argc) max_pixels = atoi(argv[++i]);
        else if (a == "--cpu-ref") cpu_ref = true;
        else if (a == "--text-check") text_check = true;
        else if (a == "--dump-ppm" && i + 1 < argc) dump = argv[++i];
        else if (a == "--wstats") {
            tensor_stats(w, "vision_tower.patchifier.weight");
            tensor_stats(w, "vision_tower.embedding_norm.weight");
            tensor_stats(w, "vision_tower.layers.0.conv.in_proj.weight");
            tensor_stats(w, "vision_tower.layers.0.conv.spatial_conv.weight");
            tensor_stats(w, "vision_tower.layers.0.conv.out_proj.weight");
            tensor_stats(w, "vision_tower.layers.0.feed_forward.w1.weight");
            tensor_stats(w, "vision_tower.layers.0.feed_forward.w2.weight");
            tensor_stats(w, "vision_tower.layers.2.self_attn.q_proj.weight");
            tensor_stats(w, "vision_tower.layers.2.self_attn.q_layernorm.weight");
            tensor_stats(w, "vision_tower.layers.13.feed_forward.w2.weight");
            tensor_stats(w, "multi_modal_projector.layer_norm.weight");
            tensor_stats(w, "multi_modal_projector.layer_norm.bias");
            tensor_stats(w, "multi_modal_projector.linear_1.weight");
            tensor_stats(w, "multi_modal_projector.linear_1.bias");
            tensor_stats(w, "multi_modal_projector.linear_2.weight");
            tensor_stats(w, "multi_modal_projector.linear_2.bias");
            return 0;
        }
    }
    std::string err;
    Vision vis(m, w);
    if (!vis.init(err)) { fprintf(stderr, "vision: %s\n", err.c_str()); return 1; }

    std::vector<uint16_t> patches;
    ImageInfo info;
    std::vector<uint8_t> rgb;
    if (!preprocess_image(image, min_pixels, max_pixels, patches, info, err, dump.empty() ? nullptr : &rgb)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    printf("image: %s\n", image.c_str());
    printf("  decoded %dx%d -> resized %dx%d (multiples of 32)\n", info.src_w, info.src_h,
           info.resized_w, info.resized_h);
    printf("  patch grid %dx%d = %d patches, %d image tokens (H/32 * W/32)\n", info.grid_w,
           info.grid_h, info.n_patches, info.n_tokens);
    if (!dump.empty()) write_ppm(dump, rgb, info.resized_w, info.resized_h);

    std::vector<uint16_t> feats;
    auto t0 = std::chrono::steady_clock::now();
    if (!vis.encode(patches, info, feats, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("vision encode: %.1f ms\n", sec * 1000.0);
    print_feature_stats(feats, info.n_tokens);

    if (text_check) {
        // The additive embedding-prefill API must reproduce the text path bit
        // for bit (multi-chunk: the prompt is > 128 tokens).
        Tokenizer tk;
        if (!tk.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Engine eng(m, w);
        if (!eng.init(err)) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
        std::string body;
        for (int i = 0; i < 40; ++i) body += "The quick brown fox jumps over the lazy dog. ";
        auto ids = tk.chat("You are a helpful assistant.", body, false);
        eng.reset();
        eng.prefill_ids(ids, true);
        std::vector<float> la(eng.logits(), eng.logits() + 128000);
        MTL::Buffer* emb = m.new_buffer(ids.size() * 2048 * 2);
        eng.embed_tokens(ids.data(), (int)ids.size(), emb);
        std::vector<uint16_t> host((size_t)ids.size() * 2048);
        memcpy(host.data(), emb->contents(), host.size() * 2);
        eng.reset();
        eng.prefill_embeds(host.data(), (int)ids.size(), true);
        std::vector<float> lb(eng.logits(), eng.logits() + 128000);
        double md = 0.0;
        int am_a = 0, am_b = 0;
        for (int i = 0; i < 128000; ++i) {
            md = std::max(md, (double)std::fabs(la[(size_t)i] - lb[(size_t)i]));
            if (la[(size_t)i] > la[(size_t)am_a]) am_a = i;
            if (lb[(size_t)i] > lb[(size_t)am_b]) am_b = i;
        }
        printf("text-check: %zu tokens (multi-chunk), max |logit(ids) - logit(embeds)| = %.6g, "
               "argmax %d/%d\n", ids.size(), md, am_a, am_b);
        if (md != 0.0 || am_a != am_b) {
            fprintf(stderr, "text-check: embedding-prefill path diverges from the token path\n");
            return 1;
        }
    }

    if (cpu_ref) {
        const int gw = 4, gh = 4;
        ImageInfo ci;
        ci.grid_w = gw;
        ci.grid_h = gh;
        ci.resized_w = gw * 16;
        ci.resized_h = gh * 16;
        ci.src_w = ci.resized_w;
        ci.src_h = ci.resized_h;
        ci.n_patches = gw * gh;
        ci.n_tokens = ci.n_patches / 4;
        auto sp = synthetic_patches(gw, gh);
        std::vector<uint16_t> gf;
        if (!vis.encode(sp, ci, gf, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::vector<float> cf;
        auto c0 = std::chrono::steady_clock::now();
        if (!vis.cpu_reference(sp, ci, cf, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        auto c1 = std::chrono::steady_clock::now();
        printf("cpu-ref on synthetic %dx%d grid (%d patches): %.1f s\n", gw, gh, ci.n_patches,
               std::chrono::duration<double>(c1 - c0).count());
        if (!compare_features(gf, cf, ci.n_tokens, err)) {
            fprintf(stderr, "VISION REFERENCE MISMATCH: %s\n", err.c_str());
            return 1;
        }
    }
    return 0;
}

static int cmd_vision_run(Metal& m, Weights& w, int argc, char** argv) {
    std::string image, prompt = "Describe this image.";
    int max_tokens = 32;
    float temp = 0.0f;
    int min_pixels = 65536, max_pixels = 16777216;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) image = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (a == "--min-pixels" && i + 1 < argc) min_pixels = atoi(argv[++i]);
        else if (a == "--max-pixels" && i + 1 < argc) max_pixels = atoi(argv[++i]);
    }
    if (image.empty()) { fprintf(stderr, "vision-run: --image <path> is required\n"); return 2; }
    std::string err;
    Tokenizer t;
    if (!t.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Vision vis(m, w);
    if (!vis.init(err)) { fprintf(stderr, "vision: %s\n", err.c_str()); return 1; }

    std::vector<uint16_t> patches;
    ImageInfo info;
    if (!preprocess_image(image, min_pixels, max_pixels, patches, info, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    printf("image: %s  %dx%d -> %dx%d, grid %dx%d, %d patches, %d image tokens\n",
           image.c_str(), info.src_w, info.src_h, info.resized_w, info.resized_h, info.grid_w,
           info.grid_h, info.n_patches, info.n_tokens);

    auto vt0 = std::chrono::steady_clock::now();
    std::vector<uint16_t> feats;
    if (!vis.encode(patches, info, feats, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto vt1 = std::chrono::steady_clock::now();
    const double vision_s = std::chrono::duration<double>(vt1 - vt0).count();
    print_feature_stats(feats, info.n_tokens);

    std::vector<int> ids = build_image_prompt(t, prompt, info.n_tokens);
    if ((int)ids.size() >= 4000) {
        fprintf(stderr, "vision-run: prompt is %zu tokens (KV capacity 4096); use --max-pixels\n",
                ids.size());
        return 1;
    }

    Engine e(m, w);
    if (!e.init(err)) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }

    MTL::Buffer* emb = m.new_buffer((size_t)ids.size() * 2048 * 2);
    e.embed_tokens(ids.data(), (int)ids.size(), emb);
    std::vector<uint16_t> host((size_t)ids.size() * 2048);
    memcpy(host.data(), emb->contents(), host.size() * 2);
    int spliced = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != kImageId) continue;
        if (spliced >= info.n_tokens) { fprintf(stderr, "vision-run: too many image tokens\n"); return 1; }
        memcpy(&host[i * 2048], &feats[(size_t)spliced * 2048], 2048 * 2);
        ++spliced;
    }
    if (spliced != info.n_tokens) {
        fprintf(stderr, "vision-run: spliced %d of %d image features\n", spliced, info.n_tokens);
        return 1;
    }
    printf("prompt: %zu tokens (%d image tokens spliced), vision %.1f ms\n", ids.size(), spliced,
           vision_s * 1000.0);

    GenConfig cfg;
    cfg.max_tokens = max_tokens;
    cfg.temperature = temp;
    e.reset();
    auto t0 = std::chrono::steady_clock::now();
    e.prefill_embeds(host.data(), (int)ids.size(), true);
    auto t1 = std::chrono::steady_clock::now();
    auto out = e.decode_loop(cfg);
    auto t2 = std::chrono::steady_clock::now();
    const double psec = std::chrono::duration<double>(t1 - t0).count();
    const double dsec = std::chrono::duration<double>(t2 - t1).count();
    printf("prefill: %zu tok %.3fs (%.0f tok/s) | decode: %zu tok %.3fs (%.1f tok/s)\n",
           ids.size(), psec, ids.size() / (psec + 1e-9), out.size(), dsec,
           out.size() / (dsec + 1e-9));
    printf("--- tokens ---\n");
    for (int id : out) printf("%d ", id);
    printf("\n--- output ---\n%s\n", t.decode(out).c_str());
    return 0;
}

int main(int argc, char** argv) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    std::string cmd = argc > 1 ? argv[1] : "info";
    std::string model = "models/model.safetensors";
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--model") model = argv[i + 1];

    Metal m;
    std::string err;
    if (!m.init("build/miori.metallib", err)) {
        fprintf(stderr, "metal init: %s\n", err.c_str());
        return 1;
    }
    if (cmd == "info") {
        m.print_info();
        printf("max threads/tg: %lu\n", (unsigned long)m.dev->maxThreadsPerThreadgroup().width);
    }

    Weights w;
    if (!w.load(m.dev, model, err)) {
        fprintf(stderr, "weights: %s\n", err.c_str());
        return 1;
    }

    if (cmd == "info" || cmd == "probe") {
        printf("tensors: %zu\n", w.tensor_count());
        const char* names[] = {
            "language_model.model.embed_tokens.weight",
            "language_model.model.layers.0.conv.in_proj.weight",
            "language_model.model.layers.2.self_attn.q_proj.weight",
            "language_model.model.layers.2.self_attn.k_norm.weight",
            "multi_modal_projector.linear_1.weight",
        };
        for (const char* n : names) {
            if (const Tensor* t = w.find(n)) {
                printf("%-60s %-4s [", n, t->dtype.c_str());
                for (size_t i = 0; i < t->shape.size(); ++i)
                    printf("%s%lld", i ? "," : "", (long long)t->shape[i]);
                printf("] %zu B\n", t->bytes());
            } else {
                printf("%-60s MISSING\n", n);
            }
        }
    } else if (cmd == "test-gemv4") {
        test_gemv(m, w, false);
    } else if (cmd == "test-gemv8") {
        test_gemv(m, w, true);
    } else if (cmd == "test-gemv4x4") {
        test_gemv(m, w, false, true);
    } else if (cmd == "test-attn") {
        const uint32_t max_seq = 8, pos = 3, kvh = 4;
        MTL::Buffer* qb = m.new_buffer(2048 * 2);
        MTL::Buffer* kb = m.new_buffer((size_t)max_seq * kvh * 128 * 2);
        MTL::Buffer* vb = m.new_buffer((size_t)kvh * 128 * max_seq * 2);
        MTL::Buffer* ob = m.new_buffer(2048 * 4);
        uint16_t* qp = (uint16_t*)qb->contents();
        uint16_t* kp = (uint16_t*)kb->contents();
        uint16_t* vp = (uint16_t*)vb->contents();
        for (int i = 0; i < 2048; ++i) qp[i] = float_to_bf16(sinf(i * 0.1f));
        for (size_t i = 0; i < (size_t)max_seq * kvh * 128; ++i) kp[i] = float_to_bf16(cosf(i * 0.07f));
        for (size_t i = 0; i < (size_t)kvh * 128 * max_seq; ++i) vp[i] = float_to_bf16(sinf(i * 0.03f));
        auto cb = m.begin();
        auto e = cb->computeCommandEncoder();
        attn_decode(m, e, qb, kb, vb, ob, pos, max_seq, kvh, 0);
        e->endEncoding();
        m.submit(cb, true);
        const float* og = (const float*)ob->contents();
        double maxerr = 0;
        for (uint32_t h = 0; h < kvh; ++h) {
            for (uint32_t w = 0; w < 4; ++w) {
                uint32_t qh = h * 4 + w;
                float sc[8];
                float mx = -1e30f;
                for (uint32_t t = 0; t <= pos; ++t) {
                    float s = 0;
                    for (uint32_t d = 0; d < 128; ++d)
                        s += bf16_to_float(qp[qh * 128 + d]) *
                             bf16_to_float(kp[((size_t)t * kvh + h) * 128 + d]);
                    sc[t] = s * 0.08838834764831845f;
                    mx = std::max(mx, sc[t]);
                }
                float sum = 0;
                for (uint32_t t = 0; t <= pos; ++t) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                for (uint32_t d = 0; d < 128; ++d) {
                    float a = 0;
                    for (uint32_t t = 0; t <= pos; ++t)
                        a += (sc[t] / sum) * bf16_to_float(vp[((size_t)h * 128 + d) * max_seq + t]);
                    maxerr = std::max(maxerr, (double)fabsf(a - og[qh * 128 + d]));
                }
            }
        }
        printf("attn max abs err = %.6g\n", maxerr);
    } else if (cmd == "test-gemm") {
        const uint32_t M = 4, N = 8, K = 32, lda = 64, ldb = 32, ldc = 16;
        MTL::Buffer* ab = m.new_buffer((size_t)M * lda * 2);
        MTL::Buffer* bb = m.new_buffer((size_t)N * ldb * 2);
        MTL::Buffer* cb = m.new_buffer((size_t)M * ldc * 4);
        uint16_t* ap = (uint16_t*)ab->contents();
        uint16_t* bp = (uint16_t*)bb->contents();
        for (uint32_t i = 0; i < M * lda; ++i) ap[i] = float_to_bf16(sinf(i * 0.13f));
        for (uint32_t i = 0; i < N * ldb; ++i) bp[i] = float_to_bf16(cosf(i * 0.17f));
        auto cb0 = m.begin();
        auto e = cb0->computeCommandEncoder();
        gemm_naive(m, e, ab, bb, cb, M, N, K, lda, ldb, ldc);
        e->endEncoding();
        m.submit(cb0, true);
        const float* cp = (const float*)cb->contents();
        double maxerr = 0;
        for (uint32_t mm = 0; mm < M; ++mm) {
            for (uint32_t nn = 0; nn < N; ++nn) {
                float ref = 0;
                for (uint32_t kk = 0; kk < K; ++kk)
                    ref += bf16_to_float(ap[mm * lda + kk]) * bf16_to_float(bp[nn * ldb + kk]);
                maxerr = std::max(maxerr, (double)fabsf(ref - cp[mm * ldc + nn]));
            }
        }
        printf("gemm max abs err = %.6g\n", maxerr);
    } else if (cmd == "test-prefill-attn") {
        const uint32_t T = 4, seq = 4, max_seq = 8, kvh = 4;
        MTL::Buffer* qb = m.new_buffer((size_t)T * 2048 * 2);
        MTL::Buffer* kb = m.new_buffer((size_t)kvh * max_seq * 128 * 2);
        MTL::Buffer* vb = m.new_buffer((size_t)kvh * 128 * max_seq * 2);
        MTL::Buffer* sb = m.new_buffer((size_t)T * seq * 4);
        MTL::Buffer* sbf = m.new_buffer((size_t)T * seq * 2);
        MTL::Buffer* ob = m.new_buffer((size_t)T * 2048 * 4);
        uint16_t* qp = (uint16_t*)qb->contents();
        uint16_t* kp = (uint16_t*)kb->contents();
        uint16_t* vp = (uint16_t*)vb->contents();
        for (size_t i = 0; i < (size_t)T * 2048; ++i) qp[i] = float_to_bf16(sinf(i * 0.01f));
        for (size_t i = 0; i < (size_t)kvh * max_seq * 128; ++i) kp[i] = float_to_bf16(cosf(i * 0.013f));
        for (size_t i = 0; i < (size_t)kvh * 128 * max_seq; ++i) vp[i] = float_to_bf16(sinf(i * 0.017f));
        auto cb = m.begin();
        auto e = cb->computeCommandEncoder();
        for (uint32_t h = 0; h < 16; ++h) {
            uint32_t kh = h / 4;
            gemm_naive(m, e, qb, kb, sb, T, seq, 128, 2048, 128, seq,
                       h * 128 * 2, kh * max_seq * 128 * 2, 0);
            softmax_prefill(m, e, sb, sbf, T, seq, 0, false);
            gemm_naive(m, e, sbf, vb, ob, T, 128, seq, seq, max_seq, 2048,
                       0, kh * 128 * max_seq * 2, h * 128 * 4);
        }
        e->endEncoding();
        m.submit(cb, true);
        const float* og = (const float*)ob->contents();
        double maxerr = 0;
        for (uint32_t h = 0; h < 16; ++h) {
            uint32_t kh = h / 4;
            for (uint32_t t = 0; t < T; ++t) {
                float sc[8], mx = -1e30f;
                for (uint32_t s = 0; s <= t; ++s) {
                    float dot = 0;
                    for (uint32_t d = 0; d < 128; ++d)
                        dot += bf16_to_float(qp[(t * 16 + h) * 128 + d]) *
                               bf16_to_float(kp[((size_t)kh * max_seq + s) * 128 + d]);
                    sc[s] = dot * 0.08838834764831845f;
                    mx = std::max(mx, sc[s]);
                }
                float sum = 0;
                for (uint32_t s = 0; s <= t; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
                for (uint32_t d = 0; d < 128; ++d) {
                    float a = 0;
                    for (uint32_t s = 0; s <= t; ++s)
                        a += (sc[s] / sum) * bf16_to_float(vp[((size_t)kh * 128 + d) * max_seq + s]);
                    maxerr = std::max(maxerr, (double)fabsf(a - og[(t * 2048) + h * 128 + d]));
                }
            }
        }
        // Re-run QK for h=15 and compare raw scores.
        {
            auto cb2 = m.begin();
            auto e2 = cb2->computeCommandEncoder();
            gemm_naive(m, e2, qb, kb, sb, T, seq, 128, 2048, 128, seq,
                       15 * 128 * 2, 3 * max_seq * 128 * 2, 0);
            e2->endEncoding();
            m.submit(cb2, true);
            const float* sg = (const float*)sb->contents();
            double serr = 0;
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t s = 0; s <= t; ++s) {
                    float dot = 0;
                    for (uint32_t d = 0; d < 128; ++d)
                        dot += bf16_to_float(qp[(t * 16 + 15) * 128 + d]) *
                               bf16_to_float(kp[((size_t)3 * max_seq + s) * 128 + d]);
                    serr = std::max(serr, (double)fabsf(dot - sg[t * seq + s]));
                }
            printf("QK raw score err = %.6g\n", serr);
        }
        double perr = 0;
        {
            const uint32_t h = 15, kh = 3;
            const uint16_t* pb = (const uint16_t*)sbf->contents();
            for (uint32_t t = 0; t < T; ++t) {
                float sc[8], mx = -1e30f;
                for (uint32_t s = 0; s <= t; ++s) {
                    float dot = 0;
                    for (uint32_t d = 0; d < 128; ++d)
                        dot += bf16_to_float(qp[(t * 16 + h) * 128 + d]) *
                               bf16_to_float(kp[((size_t)kh * max_seq + s) * 128 + d]);
                    sc[s] = dot * 0.08838834764831845f;
                    mx = std::max(mx, sc[s]);
                }
                float sum = 0;
                for (uint32_t s = 0; s <= t; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
                for (uint32_t s = 0; s <= t; ++s) {
                    const float got = bf16_to_float(pb[t * seq + s]);
                    if (t < 2) printf("  t=%u s=%u ref=%.4f got=%.4f\n", t, s, sc[s] / sum, got);
                    perr = std::max(perr, (double)fabsf(sc[s] / sum - got));
                }
            }
        }
        printf("prefill attn max abs err = %.6g, p err = %.6g\n", maxerr, perr);
    } else if (cmd == "test-conv") {
        const uint32_t T = 4, H = 2048;
        MTL::Buffer* gb = m.new_buffer(T * H * 4);
        MTL::Buffer* bb = m.new_buffer(T * 3 * H * 4);
        MTL::Buffer* sb = m.new_buffer(6 * H * 4);
        MTL::Buffer* so = m.new_buffer(6 * H * 4);
        MTL::Buffer* yb = m.new_buffer(T * H * 4);
        float* gp = (float*)gb->contents();
        float* bp = (float*)bb->contents();
        float* sp = (float*)sb->contents();
        for (uint32_t i = 0; i < T * H; ++i) gp[i] = sinf(i * 0.3f);
        for (uint32_t i = 0; i < T * 3 * H; ++i) bp[i] = cosf(i * 0.1f);
        for (uint32_t i = 0; i < 6 * H; ++i) sp[i] = 0.05f * i;
        const Tensor& kt = w.get("language_model.model.layers.0.conv.conv.weight");
        const uint16_t* wp = (const uint16_t*)kt.ptr;
        auto cb = m.begin();
        auto e = cb->computeCommandEncoder();
        conv7(m, e, gb, bb, w, kt, sb, so, yb, T, H);
        e->endEncoding();
        m.submit(cb, true);
        const float* y = (const float*)yb->contents();
        const float* so_p = (const float*)so->contents();
        double maxerr = 0;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t c = 0; c < H; ++c) {
                float acc = 0;
                for (uint32_t k = 0; k < 7; ++k) {
                    int pos = (int)t + (int)k - 6;
                    float v = pos < 0 ? sp[(uint32_t)(pos + 6) * H + c] : gp[(uint32_t)pos * H + c];
                    acc += v * bf16_to_float(wp[c * 7 + k]);
                }
                float ref = bp[t * 3 * H + H + c] * acc;
                maxerr = std::max(maxerr, (double)fabsf(ref - y[t * H + c]));
            }
        }
        double serr = 0;
        for (uint32_t k = 0; k < 6; ++k) {
            for (uint32_t c = 0; c < H; ++c) {
                int idx = (int)T - 6 + (int)k;
                float ref = idx < 0 ? sp[(uint32_t)(idx + 6) * H + c] : gp[(uint32_t)idx * H + c];
                serr = std::max(serr, (double)fabsf(ref - so_p[k * H + c]));
            }
        }
        printf("conv max abs err = %.6g, state err = %.6g\n", maxerr, serr);
    } else if (cmd == "test-tokenizer") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) {
            fprintf(stderr, "tokenizer: %s\n", err.c_str());
            return 1;
        }
        std::string text = "Hello, world! 你好，世界。 123456";
        auto ids = t.encode(text);
        printf("text: %s\nids:", text.c_str());
        for (int id : ids) printf(" %d", id);
        printf("\ndecoded: %s\n", t.decode(ids).c_str());
        auto chat = t.chat("You are a helpful assistant.", "What is 2+2?", false);
        printf("chat ids (%zu):", chat.size());
        for (size_t i = 0; i < chat.size() && i < 40; ++i) printf(" %d", chat[i]);
        printf("\nchat text: %s\n", t.decode(chat).c_str());
    } else if (cmd == "vision-test") {
        return cmd_vision_test(m, w, argc, argv);
    } else if (cmd == "vision-run") {
        return cmd_vision_run(m, w, argc, argv);
    } else if (cmd == "serve") {
        int port = 8080;
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == "--port") port = atoi(argv[i + 1]);
        return run_server(m, w, port);
    } else if (cmd == "prefix-test") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::string sys;
        for (int i = 0; i < 40; ++i)
            sys += "You are a careful assistant. Answer concisely and cite the relevant fact. ";
        std::vector<int> prefix{124894, 124899};
        auto a = t.encode("system\n" + sys);
        prefix.insert(prefix.end(), a.begin(), a.end());
        prefix.push_back(124900);
        auto nl = t.encode("\n");
        prefix.insert(prefix.end(), nl.begin(), nl.end());
        std::vector<int> suffix{124899};
        auto b = t.encode("user\nWhat is 2+2?");
        suffix.insert(suffix.end(), b.begin(), b.end());
        suffix.push_back(124900);
        suffix.insert(suffix.end(), nl.begin(), nl.end());
        suffix.push_back(124899);
        auto c = t.encode("assistant\n");
        suffix.insert(suffix.end(), c.begin(), c.end());
        suffix.push_back(124901);
        suffix.push_back(124902);
        std::vector<int> full = prefix;
        full.insert(full.end(), suffix.begin(), suffix.end());
        GenConfig cfg;
        cfg.max_tokens = 8;
        cfg.temperature = 0.0f;
        auto t0 = std::chrono::steady_clock::now();
        auto o1 = e.generate(full, cfg);
        auto t1 = std::chrono::steady_clock::now();
        e.reset();
        e.prefill_ids(prefix, false);
        e.snapshot();
        auto t2 = std::chrono::steady_clock::now();
        e.restore();
        auto o2 = e.generate_from_state(suffix, cfg);
        auto t3 = std::chrono::steady_clock::now();
        double cold = std::chrono::duration<double>(t1 - t0).count();
        double restore = std::chrono::duration<double>(t2 - t1).count();
        double warm = std::chrono::duration<double>(t3 - t2).count();
        printf("prefix=%zu suffix=%zu\n", prefix.size(), suffix.size());
        printf("cold: %.3f s  cached restore+gen: %.3f s (snapshot %.3f s)\n", cold, warm, restore);
        printf("cold out: %s\nwarm out: %s\n", t.decode_plain(o1).c_str(), t.decode_plain(o2).c_str());
    } else if (cmd == "ppl") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::string file = "docs/README.md";
        int max_tokens = 256;
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--file") file = argv[i + 1];
            if (std::string(argv[i]) == "--tokens") max_tokens = atoi(argv[i + 1]);
        }
        std::ifstream f(file);
        std::stringstream ss;
        ss << f.rdbuf();
        auto ids = t.encode(ss.str());
        ids.insert(ids.begin(), 124894);
        if ((int)ids.size() > max_tokens) ids.resize(max_tokens);
        if (ids.size() < 2) { fprintf(stderr, "corpus too short\n"); return 1; }
        double nll = 0.0;
        for (size_t i = 1; i < ids.size(); ++i) {
            e.forward(ids[i - 1], (int)(i - 1), true);
            const float* lg = e.logits();
            float mx = lg[0];
            for (int k = 1; k < 128000; ++k) mx = std::max(mx, lg[k]);
            double sum = 0.0;
            for (int k = 0; k < 128000; ++k) sum += std::exp((double)(lg[k] - mx));
            const double cur = (double)mx + std::log(sum) - (double)lg[ids[i]];
            if (getenv("MIORI_PPL_DEBUG")) {
                int rank = 0;
                for (int k = 0; k < 128000; ++k) if (lg[k] > lg[ids[i]]) ++rank;
                fprintf(stderr, "%zu id=%d rank=%d nll=%.3f  ", i, ids[i], rank, cur);
                fprintf(stderr, "%s\n", t.token(ids[i]).c_str());
            }
            nll += cur;
        }
        printf("ppl: %.3f over %zu tokens (%s)\n",
               std::exp(nll / (double)(ids.size() - 1)), ids.size() - 1, file.c_str());
    } else if (cmd == "bench") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        int plen = 1024, dlen = 128;
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--prefill") plen = atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--decode") dlen = atoi(argv[i + 1]);
        }
        std::string text;
        while ((int)text.size() < plen * 12) text += "The quick brown fox jumps over the lazy dog. ";
        auto ids = t.encode(text);
        if ((int)ids.size() > plen) ids.resize(plen);
        GenConfig cfg;
        cfg.max_tokens = dlen;
        cfg.temperature = 0.0f;
        std::vector<double> ps, ds;
        for (int r = 0; r < 3; ++r) {
            auto out = e.generate(ids, cfg);
            ps.push_back(e.last_prefill_s());
            ds.push_back(e.last_decode_s());
        }
        std::sort(ps.begin(), ps.end());
        std::sort(ds.begin(), ds.end());
        const double p = ps[1], d = ds[1];
        printf("\nbench: prompt=%zu output=%d\n", ids.size(), dlen);
        printf("prefill: %.3f s  %.1f tok/s\n", p, ids.size() / p);
        printf("decode:  %.3f s  %.1f tok/s  (%.1f GB/s effective)\n",
               d, dlen / d, 1.212e9 * (dlen / d) / 1e9);
    } else if (cmd == "batch-test") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) {
            fprintf(stderr, "tokenizer: %s\n", err.c_str());
            return 1;
        }
        std::vector<std::string> prompts;
        int max_tokens = 16;
        BatchEngine::Kernels kernels = BatchEngine::Kernels::MppBf16;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--prompts") {
                int j = i + 1;
                while (j < argc && argv[j][0] != '-') prompts.push_back(argv[j++]);
                i = j - 1;
            } else if (a == "--max-tokens" && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
            } else if (a == "--kernels" && i + 1 < argc) {
                std::string k = argv[++i];
                if (k == "gemv4") kernels = BatchEngine::Kernels::Gemv4;
                else if (k != "mpp") {
                    fprintf(stderr, "batch-test: --kernels must be mpp or gemv4\n");
                    return 2;
                }
            }
        }
        if (prompts.empty())
            prompts = {"The capital of France is", "List the first four primes."};
        if (prompts.size() < 2) prompts.push_back("What is 2+2?");
        if ((int)prompts.size() > BatchEngine::kBmax) prompts.resize(BatchEngine::kBmax);

        Engine e(m, w);
        if (!e.init(err)) {
            fprintf(stderr, "engine: %s\n", err.c_str());
            return 1;
        }
        BatchEngine be(m, w, e);
        be.set_kernels(kernels);
        if (!be.init(err)) {
            fprintf(stderr, "batch: %s\n", err.c_str());
            return 1;
        }
        const std::string sys = "You are a helpful assistant.";
        std::vector<std::vector<int>> ids(prompts.size());
        for (size_t i = 0; i < prompts.size(); ++i) ids[i] = t.chat(sys, prompts[i], false);

        // Reference: single-engine greedy decode, one sequence at a time.
        GenConfig cfg;
        cfg.max_tokens = max_tokens;
        cfg.temperature = 0.0f;
        std::vector<std::vector<int>> ref(prompts.size());
        double ref_fwd = 0.0, ref_s = 0.0;
        for (size_t i = 0; i < prompts.size(); ++i) {
            ref[i] = e.generate(ids[i], cfg);
            ref_fwd += (double)(ref[i].size() > 1 ? ref[i].size() - 1 : 0);
            ref_s += e.last_decode_s();
        }

        // Batch: sequential prefill into slots, then one batched step per token.
        std::vector<int> slot(prompts.size(), -1);
        auto tp0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompts.size(); ++i) {
            slot[i] = be.add_sequence(ids[i], max_tokens);
            if (slot[i] < 0) {
                fprintf(stderr, "batch: add_sequence failed for prompt %zu\n", i);
                return 1;
            }
        }
        auto tp1 = std::chrono::steady_clock::now();
        const double prefill_s = std::chrono::duration<double>(tp1 - tp0).count();

        int guard = max_tokens + 8;
        while (be.running() > 0 && guard-- > 0) be.decode_step();

        int total = 0, mismatch = 0;
        for (size_t i = 0; i < prompts.size(); ++i) {
            const auto& g = be.generated(slot[i]);
            printf("seq %zu: %zu prompt tokens, %zu generated\n  ref:", i, ids[i].size(), g.size());
            for (int id : ref[i]) printf(" %d", id);
            printf("\n  bat:");
            for (int id : g) printf(" %d", id);
            printf("\n");
            const size_t n = std::max(g.size(), ref[i].size());
            for (size_t k = 0; k < n; ++k) {
                ++total;
                const int a = k < ref[i].size() ? ref[i][k] : -1;
                const int b = k < g.size() ? g[k] : -1;
                if (a != b) {
                    if (mismatch == 0)
                        printf("  first mismatch at token %zu: ref=%d bat=%d\n", k, a, b);
                    ++mismatch;
                }
            }
        }
        const double batch_s = be.total_decode_s();
        const int batch_tok = be.total_generated();
        const double agg = batch_tok / std::max(1e-9, batch_s);
        const double ref_rate = ref_s > 0.0 ? ref_fwd / ref_s : 0.0;
        printf("\nbatch-test: B=%zu (Bmax=%d), max_tokens=%d, kernels=%s\n", prompts.size(),
               BatchEngine::kBmax, max_tokens, be.kernels_name());
        printf("prefill (sequential Engine): %.3f s\n", prefill_s);
        printf("reference single-stream decode: %.0f forwards in %.3f s = %.1f tok/s\n",
               ref_fwd, ref_s, ref_rate);
        printf("batch decode: %d steps, %d tokens in %.3f s = %.1f tok/s aggregate "
               "(%.2f ms/step)\n",
               be.steps(), batch_tok, batch_s, agg,
               1000.0 * batch_s / std::max(1, be.steps()));
        printf("tokens compared: %d, mismatches: %d\n", total, mismatch);
        printf("aggregate speedup vs sequential single-stream: %.2fx\n",
               agg / std::max(1e-9, ref_rate));
        if (mismatch == 0) printf("RESULT: token-identical to single-sequence greedy\n");
        else printf("RESULT: %d/%d tokens differ (bf16 MPP GEMM vs 4-bit GEMV numerics)\n",
                    mismatch, total);
    } else if (cmd == "run") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) {
            fprintf(stderr, "tokenizer: %s\n", err.c_str());
            return 1;
        }
        Engine e(m, w);
        if (!e.init(err)) {
            fprintf(stderr, "engine: %s\n", err.c_str());
            return 1;
        }
        std::string prompt = "What is the capital of France?";
        int max_tokens = 24;
        float temp = 0.0f;
        bool raw = false, thinking = false, slow = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
            else if (a == "--max-tokens" && i + 1 < argc) max_tokens = atoi(argv[++i]);
            else if (a == "--temp" && i + 1 < argc) temp = (float)atof(argv[++i]);
            else if (a == "--raw") raw = true;
            else if (a == "--thinking") thinking = true;
            else if (a == "--slow-prefill") slow = true;
        }
        std::vector<int> ids;
        if (raw) {
            ids.push_back(124894);
            auto body = t.encode(prompt);
            ids.insert(ids.end(), body.begin(), body.end());
        } else {
            ids = t.chat("You are a helpful assistant.", prompt, thinking);
        }
        printf("prompt: %zu tokens\n%s\n", ids.size(), t.decode(ids).c_str());
        GenConfig cfg;
        cfg.max_tokens = max_tokens;
        cfg.temperature = temp;
        cfg.slow_prefill = slow;
        auto t0 = std::chrono::steady_clock::now();
        auto out = e.generate(ids, cfg);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        printf("--- output ---\n%s\n--- %zu tokens in %.3f s = %.2f tok/s ---\n",
               t.decode(out).c_str(), out.size(), sec, out.size() / sec);
    } else {
        fprintf(stderr, "unknown command: %s\n", cmd.c_str());
        return 2;
    }
    pool->release();
    return 0;
}
