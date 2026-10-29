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

// Number of differing bytes in two host-visible buffers (packed prefill checks).
static size_t diff_bytes(const void* a, const void* b, size_t n) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += (pa[i] != pb[i]) ? 1 : 0;
    return d;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

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
    int repeat = 1;
    std::string dump;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) image = argv[++i];
        else if (a == "--min-pixels" && i + 1 < argc) min_pixels = atoi(argv[++i]);
        else if (a == "--max-pixels" && i + 1 < argc) max_pixels = atoi(argv[++i]);
        else if (a == "--cpu-ref") cpu_ref = true;
        else if (a == "--text-check") text_check = true;
        else if (a == "--repeat" && i + 1 < argc) repeat = atoi(argv[++i]);
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
    double sec = 0.0;
    for (int r = 0; r < std::max(1, repeat); ++r) {
        auto t0 = std::chrono::steady_clock::now();
        if (!vis.encode(patches, info, feats, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        auto t1 = std::chrono::steady_clock::now();
        sec = std::chrono::duration<double>(t1 - t0).count();
        printf("vision encode%s: %.1f ms\n", repeat > 1 ? (r == 0 ? " (cold)" : " (warm)") : "",
               sec * 1000.0);
    }
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

static int cmd_batch_sparse(Metal& m, Weights& w) {
    Tokenizer t;
    std::string err;
    if (!t.load("models/tokenizer.json", err)) {
        fprintf(stderr, "tokenizer: %s\n", err.c_str());
        return 1;
    }
    Engine e(m, w);
    if (!e.init(err)) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
    BatchEngine be(m, w, e);
    if (!be.init(err)) { fprintf(stderr, "batch: %s\n", err.c_str()); return 1; }
    be.set_kernels(BatchEngine::Kernels::Gemv4);

    const std::string sys = "You are a helpful assistant.";
    std::vector<std::string> prompts = {
        "The capital of France is", "List the first four primes.",
        "What is 2+2?", "Name three colors."};
    GenConfig cfg;
    cfg.max_tokens = 3;
    cfg.temperature = 0.0f;
    std::vector<std::vector<int>> ids, ref;
    for (auto& p : prompts) {
        ids.push_back(t.chat(sys, p, false));
        ref.push_back(e.generate(ids.back(), cfg));
    }
    std::vector<int> slots;
    for (auto& v : ids) slots.push_back(be.add_sequence(v, 3));
    int guard = 20;
    while (be.running() > 0 && guard-- > 0) be.decode_step();
    for (size_t i = 0; i < slots.size(); ++i)
        if ((i == 1 || i == 3) && slots[i] >= 0) be.release_slot(slots[i]);

    std::vector<std::string> prompts2 = {
        "What is the capital of Italy?", "Describe the ocean in one word."};
    std::vector<std::vector<int>> ids2, ref2;
    cfg.max_tokens = 4;
    for (auto& p : prompts2) {
        ids2.push_back(t.chat(sys, p, false));
        ref2.push_back(e.generate(ids2.back(), cfg));
    }
    std::vector<int> slots2;
    for (auto& v : ids2) slots2.push_back(be.add_sequence(v, 4));
    guard = 20;
    while (be.running() > 0 && guard-- > 0) be.decode_step();

    int bad = 0;
    for (size_t i = 0; i < prompts2.size(); ++i) {
        if (slots2[i] < 0) { printf("sparse: add failed\n"); return 1; }
        const auto& g = be.generated(slots2[i]);
        bool same = g == ref2[i];
        printf("sparse %zu slot %d: %s (%zu vs %zu tokens)\n", i, slots2[i],
               same ? "identical" : "DIFF", g.size(), ref2[i].size());
        if (!same) ++bad;
    }
    printf("batch-sparse-test: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
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
        MTL::Buffer* yb = m.new_buffer(T * H * 2);
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
        const uint16_t* y = (const uint16_t*)yb->contents();
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
                maxerr = std::max(maxerr, (double)fabsf(ref - bf16_to_float(y[t * H + c])));
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
    } else if (cmd == "test-kv-int8") {
        // Kernel-level check for the optional int8 KV path: quantise K/V with
        // the Metal writers, then compare prefill QK/PV and decode attention
        // (normal + split-K) against a CPU reference that performs the same
        // quantise/dequantise arithmetic.
        const uint32_t kvh = 4, dim = 128, max_seq = 640, npos = 601;
        const size_t kv_elems = (size_t)kvh * max_seq * dim;
        MTL::Buffer* qd = m.new_buffer(16 * dim * 2);
        MTL::Buffer* kb = m.new_buffer((size_t)npos * 512 * 2);
        MTL::Buffer* vb = m.new_buffer((size_t)npos * 512 * 4);
        MTL::Buffer* kc8 = m.new_buffer(kv_elems);
        MTL::Buffer* vc8 = m.new_buffer(kv_elems);
        const size_t scale_elems = (size_t)kvh * max_seq * 4;   // 4 groups per token
        MTL::Buffer* ks8 = m.new_buffer(scale_elems * 2);
        MTL::Buffer* vs8 = m.new_buffer(scale_elems * 2);
        MTL::Buffer* outd = m.new_buffer(2048 * 4);
        MTL::Buffer* outsplit = m.new_buffer(2048 * 4);
        MTL::Buffer* pacc = m.new_buffer((size_t)kvh * 16 * 4 * 128 * 4);
        MTL::Buffer* pm = m.new_buffer((size_t)kvh * 16 * 4 * 4);
        MTL::Buffer* pl = m.new_buffer((size_t)kvh * 16 * 4 * 4);
        memset(kc8->contents(), 0, kv_elems);
        memset(vc8->contents(), 0, kv_elems);
        memset(ks8->contents(), 0, scale_elems * 2);
        memset(vs8->contents(), 0, scale_elems * 2);
        uint16_t* qp = (uint16_t*)qd->contents();
        for (uint32_t i = 0; i < 16 * dim; ++i) qp[i] = float_to_bf16(sinf(i * 0.017f));
        uint16_t* kp = (uint16_t*)kb->contents();
        float* vp = (float*)vb->contents();
        for (uint32_t p = 0; p < npos; ++p) {
            for (uint32_t r = 0; r < 512; ++r) {
                kp[(size_t)p * 512 + r] =
                    float_to_bf16(1.7f * sinf(0.0031f * (p * 512 + r)) +
                                  0.2f * cosf(0.011f * r));
                vp[(size_t)p * 512 + r] =
                    2.3f * cosf(0.0027f * (p * 512 + r)) + 0.4f * sinf(0.019f * r);
            }
        }
        {
            auto cb = m.begin();
            auto enc = cb->computeCommandEncoder();
            kv_write_b_i8(m, enc, kb, vb, kc8, vc8, ks8, vs8, npos, 0, max_seq);
            enc->endEncoding();
            m.submit(cb, true);
        }

        // CPU reference: quantise exactly like kv_write_b_i8 (per-32 group
        // scale from the unrounded max_abs/127; store the bf16-rounded scale;
        // q rounded to nearest), then dequantise on read.
        std::vector<float> kdq(kv_elems), vdq(kv_elems);
        double kq_rel = 0.0, vq_rel = 0.0;
        for (uint32_t h = 0; h < kvh; ++h) {
            for (uint32_t p = 0; p < npos; ++p) {
                for (uint32_t g = 0; g < 4; ++g) {
                    float kmax = 0.0f, vmax = 0.0f;
                    for (uint32_t d = g * 32; d < (g + 1) * 32; ++d) {
                        kmax = std::max(kmax, fabsf(bf16_to_float(
                                                    kp[(size_t)p * 512 + h * 128 + d])));
                        vmax = std::max(vmax,
                                        fabsf(vp[(size_t)p * 512 + h * 128 + d]));
                    }
                    const float ks = kmax > 0.0f ? kmax / 127.0f : 0.0f;
                    const float vs = vmax > 0.0f ? vmax / 127.0f : 0.0f;
                    const float ksb = bf16_to_float(float_to_bf16(ks));
                    const float vsb = bf16_to_float(float_to_bf16(vs));
                    for (uint32_t d = g * 32; d < (g + 1) * 32; ++d) {
                        const float kf = bf16_to_float(kp[(size_t)p * 512 + h * 128 + d]);
                        const float vf = vp[(size_t)p * 512 + h * 128 + d];
                        const int kqi = ks > 0.0f
                                            ? (int)lrintf(std::max(-127.0f, std::min(127.0f, kf / ks)))
                                            : 0;
                        const int vqi = vs > 0.0f
                                            ? (int)lrintf(std::max(-127.0f, std::min(127.0f, vf / vs)))
                                            : 0;
                        kdq[((size_t)h * max_seq + p) * dim + d] = (float)kqi * ksb;
                        vdq[((size_t)h * dim + d) * max_seq + p] = (float)vqi * vsb;
                        if (kmax > 0.0f)
                            kq_rel = std::max(
                                kq_rel, (double)(fabsf((float)kqi * ksb - kf) / kmax));
                        if (vmax > 0.0f)
                            vq_rel = std::max(
                                vq_rel, (double)(fabsf((float)vqi * vsb - vf) / vmax));
                    }
                }
            }
        }
        auto kref = [&](uint32_t h, uint32_t p, uint32_t d) {
            return kdq[((size_t)h * max_seq + p) * dim + d];
        };
        auto vref = [&](uint32_t h, uint32_t d, uint32_t p) {
            return vdq[((size_t)h * dim + d) * max_seq + p];
        };
        const float qscale = 0.08838834764831845f;

        // ---- prefill QK + softmax + PV for T=4 tokens at positions 0..3
        {
            const uint32_t T = 4, seq = 4;
            MTL::Buffer* pqn = m.new_buffer((size_t)T * 2048 * 2);
            MTL::Buffer* pv = m.new_buffer((size_t)T * 512 * 4);
            MTL::Buffer* pscores = m.new_buffer((size_t)T * seq * 4);
            MTL::Buffer* psbf = m.new_buffer((size_t)T * seq * 2);
            MTL::Buffer* pattn = m.new_buffer((size_t)T * 2048 * 4);
            uint16_t* pqq = (uint16_t*)pqn->contents();
            float* pvv = (float*)pv->contents();
            for (uint32_t i = 0; i < T * 2048; ++i)
                pqq[i] = float_to_bf16(cosf(i * 0.013f) * 0.9f);
            for (uint32_t i = 0; i < T * 512; ++i) pvv[i] = sinf(i * 0.021f);
            memset(pscores->contents(), 0, (size_t)T * seq * 4);
            memset(psbf->contents(), 0, (size_t)T * seq * 2);
            memset(pattn->contents(), 0, (size_t)T * 2048 * 4);
            auto cb = m.begin();
            auto enc = cb->computeCommandEncoder();
            for (uint32_t h = 0; h < 16; ++h) {
                const uint32_t khead = h / 4;
                gemm_naive_qk_i8(m, enc, pqn, kc8, ks8, pscores, T, seq, 128, 2048, dim,
                                 seq, h * 128 * 2, khead * max_seq * dim * 1u,
                                 khead * max_seq * 4u * 2u, 0);
                softmax_prefill(m, enc, pscores, psbf, T, seq, 0, false);
                gemm_naive_pv_i8(m, enc, psbf, vc8, vs8, pattn, T, 128, seq, seq, max_seq,
                                 2048, 0, khead * dim * max_seq,
                                 khead * max_seq * 4u * 2u, h * 128 * 4);
            }
            enc->endEncoding();
            m.submit(cb, true);
            const float* pg = (const float*)pattn->contents();
            double perr = 0.0;
            for (uint32_t h = 0; h < 16; ++h) {
                for (uint32_t tt = 0; tt < T; ++tt) {
                    float sc[8], mx = -1e30f;
                    for (uint32_t ss = 0; ss <= tt; ++ss) {
                        float dot = 0.0f;
                        for (uint32_t d = 0; d < dim; ++d)
                            dot += bf16_to_float(pqq[(tt * 16 + h) * 128 + d]) *
                                   kref(h / 4, ss, d);
                        sc[ss] = dot * qscale;
                        mx = std::max(mx, sc[ss]);
                    }
                    float sum = 0.0f;
                    for (uint32_t ss = 0; ss <= tt; ++ss) {
                        sc[ss] = expf(sc[ss] - mx);
                        sum += sc[ss];
                    }
                    for (uint32_t d = 0; d < dim; ++d) {
                        float a = 0.0f;
                        for (uint32_t ss = 0; ss <= tt; ++ss) {
                            // softmax_prefill stores probabilities as bf16, so the
                            // CPU PV reference has to round them the same way.
                            const float pb = bf16_to_float(float_to_bf16(sc[ss] / sum));
                            a += pb * vref(h / 4, d, ss);
                        }
                        perr = std::max(perr, (double)fabsf(a - pg[(tt * 2048) + h * 128 + d]));
                    }
                }
            }
            // Raw QK scores, re-run for one head after the main dispatch.
            double qkerr = 0.0;
            {
                auto cb2 = m.begin();
                auto enc2 = cb2->computeCommandEncoder();
                gemm_naive_qk_i8(m, enc2, pqn, kc8, ks8, pscores, T, seq, 128, 2048, dim,
                                 seq, 15 * 128 * 2, 3 * max_seq * dim,
                                 3 * max_seq * 4u * 2u, 0);
                enc2->endEncoding();
                m.submit(cb2, true);
                const float* sg = (const float*)pscores->contents();
                for (uint32_t tt = 0; tt < T; ++tt)
                    for (uint32_t ss = 0; ss <= tt; ++ss) {
                        float dot = 0.0f;
                        for (uint32_t d = 0; d < dim; ++d)
                            dot += bf16_to_float(pqq[(tt * 16 + 15) * 128 + d]) *
                                   kref(3, ss, d);
                        qkerr = std::max(qkerr, (double)fabsf(dot - sg[tt * seq + ss]));
                    }
            }
            printf("test-kv-int8: prefill (T=4, seq=4) QK raw score err = %.6g, "
                   "PV max abs err = %.6g\n", qkerr, perr);
        }

        // ---- decode attention at pos=600, full and split-K
        const uint32_t pos = npos - 1;
        double derr = 0.0, serr = 0.0;
        {
            auto cb = m.begin();
            auto enc = cb->computeCommandEncoder();
            attn_decode_i8(m, enc, qd, kc8, vc8, ks8, vs8, outd, pos, max_seq, kvh, 0);
            attn_decode_split_i8(m, enc, qd, kc8, vc8, ks8, vs8, pacc, pm, pl, pos, max_seq,
                                 kvh, 16);
            attn_combine(m, enc, pacc, pm, pl, outsplit, 16);
            enc->endEncoding();
            m.submit(cb, true);
        }
        const float* og = (const float*)outd->contents();
        const float* sg = (const float*)outsplit->contents();
        for (uint32_t h = 0; h < kvh; ++h) {
            for (uint32_t w = 0; w < 4; ++w) {
                const uint32_t qh = h * 4 + w;
                float sc[601], mx = -1e30f;
                for (uint32_t tt = 0; tt <= pos; ++tt) {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d)
                        dot += bf16_to_float(qp[qh * 128 + d]) * kref(h, tt, d);
                    sc[tt] = dot * qscale;
                    mx = std::max(mx, sc[tt]);
                }
                float sum = 0.0f;
                for (uint32_t tt = 0; tt <= pos; ++tt) {
                    sc[tt] = expf(sc[tt] - mx);
                    sum += sc[tt];
                }
                for (uint32_t d = 0; d < dim; ++d) {
                    float a = 0.0f;
                    for (uint32_t tt = 0; tt <= pos; ++tt)
                        a += (sc[tt] / sum) * vref(h, d, tt);
                    derr = std::max(derr, (double)fabsf(a - og[qh * 128 + d]));
                    serr = std::max(serr, (double)fabsf(a - sg[qh * 128 + d]));
                }
            }
        }
        printf("test-kv-int8: decode (pos=%u) max abs err = %.6g, split-K(16) = %.6g\n", pos,
               derr, serr);
        // ---- mixed K/V decode kernels: one side reads the bf16 cache below.
        MTL::Buffer* kbf = m.new_buffer(kv_elems * 2);
        MTL::Buffer* vbf = m.new_buffer(kv_elems * 2);
        uint16_t* kbfp = (uint16_t*)kbf->contents();
        uint16_t* vbfp = (uint16_t*)vbf->contents();
        for (uint32_t h = 0; h < kvh; ++h) {
            for (uint32_t p2 = 0; p2 < npos; ++p2) {
                for (uint32_t d = 0; d < dim; ++d) {
                    const uint32_t r = h * 128u + d;
                    kbfp[((size_t)h * max_seq + p2) * dim + d] =
                        kp[(size_t)p2 * 512 + r];
                    vbfp[((size_t)h * dim + d) * max_seq + p2] =
                        float_to_bf16(vp[(size_t)p2 * 512 + r]);
                }
            }
        }
        MTL::Buffer* out_kv = m.new_buffer(2048 * 4);
        MTL::Buffer* out_vk = m.new_buffer(2048 * 4);
        {
            auto cb = m.begin();
            auto enc = cb->computeCommandEncoder();
            // mix=vk: K bf16 + V int8 (k_int8 = 0)
            attn_decode_mix(m, enc, qd, kbf, kc8, ks8, kbf, vc8, vs8, out_vk, pos,
                            max_seq, kvh, 0, 0);
            // mix=kv: K int8 + V bf16 (k_int8 = 1)
            attn_decode_mix(m, enc, qd, kbf, kc8, ks8, vbf, vc8, vs8, out_kv, pos,
                            max_seq, kvh, 0, 1);
            enc->endEncoding();
            m.submit(cb, true);
        }
        auto mixed_ref = [&](bool kbf_side, bool vbf_side, const float* got,
                             double& err) {
            for (uint32_t h = 0; h < kvh; ++h) {
                for (uint32_t w = 0; w < 4; ++w) {
                    const uint32_t qh = h * 4 + w;
                    float sc[601], mx = -1e30f;
                    for (uint32_t tt = 0; tt <= pos; ++tt) {
                        float dot = 0.0f;
                        for (uint32_t d = 0; d < dim; ++d) {
                            const float kk = kbf_side
                                ? bf16_to_float(kbfp[((size_t)h * max_seq + tt) * dim + d])
                                : kref(h, tt, d);
                            dot += bf16_to_float(qp[qh * 128 + d]) * kk;
                        }
                        sc[tt] = dot * qscale;
                        mx = std::max(mx, sc[tt]);
                    }
                    float sum = 0.0f;
                    for (uint32_t tt = 0; tt <= pos; ++tt) {
                        sc[tt] = expf(sc[tt] - mx);
                        sum += sc[tt];
                    }
                    for (uint32_t d = 0; d < dim; ++d) {
                        float a = 0.0f;
                        for (uint32_t tt = 0; tt <= pos; ++tt) {
                            const float vv = vbf_side
                                ? bf16_to_float(vbfp[((size_t)h * dim + d) * max_seq + tt])
                                : vref(h, d, tt);
                            a += (sc[tt] / sum) * vv;
                        }
                        err = std::max(err, (double)fabsf(a - got[qh * 128 + d]));
                    }
                }
            }
        };
        double mix_kv_err = 0.0, mix_vk_err = 0.0;
        mixed_ref(false, true, (const float*)out_kv->contents(), mix_kv_err);
        mixed_ref(true, false, (const float*)out_vk->contents(), mix_vk_err);
        printf("test-kv-int8: mixed decode K int8/V bf16 err = %.6g, K bf16/V int8 err = "
               "%.6g\n", mix_kv_err, mix_vk_err);
        printf("test-kv-int8: max quantisation error / 32-group max_abs: K %.3f%%, V %.3f%% "
               "(1/127 = %.2f%%)\n",
               100.0 * kq_rel, 100.0 * vq_rel, 100.0 / 127.0);
        const bool ok = derr < 5e-4 && serr < 5e-4 && mix_kv_err < 5e-4 &&
                        mix_vk_err < 5e-4;
        printf("test-kv-int8: %s\n", ok ? "PASS" : "FAIL");
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
    } else if (cmd == "kv-int8-test") {
        // Quality gate for the optional int8 KV cache: same 1K corpus through
        // the bf16 and int8 paths, then PPL, top-1 agreement and full-vocab KL
        // at a sample of positions. Optionally measure 4K decode throughput.
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::string file = "docs/README.md";
        int max_tokens = 1024, decode_len = 32, ctx = 4096;
        bool do_bench = true;
        Engine::KvMode cand_mode = Engine::KvMode::Int8;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--file" && i + 1 < argc) file = argv[++i];
            else if (a == "--tokens" && i + 1 < argc) max_tokens = atoi(argv[++i]);
            else if (a == "--decode" && i + 1 < argc) decode_len = atoi(argv[++i]);
            else if (a == "--ctx" && i + 1 < argc) ctx = atoi(argv[++i]);
            else if (a == "--no-bench") do_bench = false;
            else if (a == "--mix" && i + 1 < argc) {
                std::string v = argv[++i];
                if (v == "kv") cand_mode = Engine::KvMode::K8V16;
                else if (v == "vk") cand_mode = Engine::KvMode::K16V8;
                else { fprintf(stderr, "kv-int8-test: --mix must be kv or vk\n"); return 2; }
            }
        }
        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
        // The comparison always runs bf16 first, regardless of the env flag.
        if (!e.set_kv_mode(Engine::KvMode::Bf16, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const size_t kv_bf16_bytes = e.kv_cache_bytes();

        std::ifstream f(file);
        if (!f) { fprintf(stderr, "kv-int8-test: cannot open %s\n", file.c_str()); return 1; }
        std::stringstream ss;
        ss << f.rdbuf();
        auto ids = t.encode(ss.str());
        ids.insert(ids.begin(), 124894);
        if ((int)ids.size() > max_tokens) ids.resize(max_tokens);
        if (ids.size() < 3) { fprintf(stderr, "kv-int8-test: corpus too short\n"); return 1; }
        const size_t N = ids.size();
        const int nSamples = (int)std::min<size_t>(128, N - 1);
        std::vector<int> sample_index(N, -1);
        std::vector<size_t> sample_at;
        for (int k = 0; k < nSamples; ++k) {
            const size_t i = nSamples > 1 ? 1 + (size_t)k * (N - 2) / (size_t)(nSamples - 1)
                                          : (size_t)1;
            if (sample_index[i] >= 0) continue;
            sample_index[i] = (int)sample_at.size();
            sample_at.push_back(i);
        }
        const int nS = (int)sample_at.size();

        struct PassStats {
            std::vector<int> top1;
            std::vector<double> nll;
            double ppl = 0.0;
        };
        auto run_pass = [&](Engine::KvMode m, PassStats& st,
                            std::vector<float>* sampled_out,
                            const std::vector<float>* ref_sampled,
                            std::vector<double>* kls, double* dlogit_sum) -> bool {
            if (!e.set_kv_mode(m, err)) {
                fprintf(stderr, "kv-int8-test: %s\n", err.c_str());
                return false;
            }
            e.reset();
            st.top1.assign(N, -1);
            st.nll.assign(N - 1, 0.0);
            if (sampled_out) sampled_out->assign((size_t)nS * 128000, 0.0f);
            if (kls) { kls->clear(); kls->reserve((size_t)nS); }
            if (dlogit_sum) *dlogit_sum = 0.0;
            for (size_t i = 1; i < N; ++i) {
                e.forward(ids[i - 1], (int)(i - 1), true);
                const float* lg = e.logits();
                int am = 0;
                float mx = lg[0];
                for (int v = 1; v < 128000; ++v)
                    if (lg[v] > mx) { mx = lg[v]; am = v; }
                st.top1[i] = am;
                double sum = 0.0;
                for (int v = 0; v < 128000; ++v) sum += std::exp((double)lg[v] - mx);
                st.nll[i - 1] = (double)mx + std::log(sum) - (double)lg[ids[i]];
                const int si = sample_index[i];
                if (si < 0) continue;
                if (sampled_out)
                    memcpy(&(*sampled_out)[(size_t)si * 128000], lg, (size_t)128000 * 4);
                if (ref_sampled && kls) {
                    const float* r = ref_sampled->data() + (size_t)si * 128000;
                    double rmx = r[0];
                    for (int v = 1; v < 128000; ++v) rmx = std::max(rmx, (double)r[v]);
                    double rsum = 0.0, csum = 0.0;
                    for (int v = 0; v < 128000; ++v) {
                        rsum += std::exp((double)r[v] - rmx);
                        csum += std::exp((double)lg[v] - mx);
                    }
                    const double rlse = rmx + std::log(rsum);
                    const double clse = (double)mx + std::log(csum);
                    double kl = 0.0;
                    for (int v = 0; v < 128000; ++v) {
                        const double p = std::exp((double)r[v] - rmx) / rsum;
                        kl += p * (((double)r[v] - rlse) - ((double)lg[v] - clse));
                    }
                    kls->push_back(kl);
                    if (dlogit_sum)
                        for (int v = 0; v < 128000; ++v)
                            *dlogit_sum += std::fabs((double)r[v] - (double)lg[v]);
                }
            }
            double nll_sum = 0.0;
            for (double v : st.nll) nll_sum += v;
            st.ppl = std::exp(nll_sum / (double)st.nll.size());
            return true;
        };

        std::vector<float> ref_sampled;
        PassStats ref, cand;
        if (!run_pass(Engine::KvMode::Bf16, ref, &ref_sampled, nullptr, nullptr, nullptr))
            return 1;
        std::vector<double> kls;
        double dlogit_sum = 0.0;
        if (!run_pass(cand_mode, cand, nullptr, &ref_sampled, &kls, &dlogit_sum)) return 1;
        const size_t kv_cand_bytes = e.kv_cache_bytes();

        int agree = 0;
        for (size_t i = 1; i < N; ++i)
            if (ref.top1[i] == cand.top1[i]) ++agree;
        const double agree_pct = 100.0 * (double)agree / (double)(N - 1);
        std::sort(kls.begin(), kls.end());
        double kl_mean = 0.0;
        for (double v : kls) kl_mean += v;
        kl_mean /= (double)kls.size();
        const double kl_p99 = kls.empty() ? 0.0 : kls[(size_t)std::min<size_t>(kls.size() - 1,
                                                        (size_t)std::floor(0.99 * kls.size()))];
        const double kl_max = kls.empty() ? 0.0 : kls.back();
        const double dlogit = dlogit_sum / ((double)nS * 128000.0);
        const double ppl_delta = 100.0 * (cand.ppl / ref.ppl - 1.0);

        // Modelled native KV footprint (BatchEngine, all 32 slots) and per-token
        // attention read traffic at 4K context.
        auto slot_bytes = [](Engine::KvMode m) {
            const double k = Engine::kv_mode_k_int8(m)
                                 ? 4.0 * 4096.0 * 128.0 + 4.0 * 4096.0 * 4.0 * 2.0
                                 : 4.0 * 4096.0 * 128.0 * 2.0;
            const double v = Engine::kv_mode_v_int8(m)
                                 ? 4.0 * 4096.0 * 128.0 + 4.0 * 4096.0 * 4.0 * 2.0
                                 : 4.0 * 4096.0 * 128.0 * 2.0;
            return 11.0 * (k + v);
        };
        auto kv_tok_bytes = [](Engine::KvMode m) {
            const double k = Engine::kv_mode_k_int8(m) ? 1.0 : 2.0;
            const double v = Engine::kv_mode_v_int8(m) ? 1.0 : 2.0;
            return 11.0 * 16.0 * 4096.0 * 128.0 * (k + v);
        };
        const double slot_bf16 = slot_bytes(Engine::KvMode::Bf16);
        const double slot_cand = slot_bytes(cand_mode);

        printf("\nkv-int8-test: %s, %zu positions (%d KL sample(s)), candidate %s\n",
               file.c_str(), N - 1, nS, Engine::kv_mode_name(cand_mode));
        printf("KV bytes: Engine %.2f MB -> %.2f MB (%.1f%%); modelled BatchEngine Bmax=32 "
               "%.1f MB -> %.1f MB (%.1f%%)\n",
               (double)kv_bf16_bytes / 1e6, (double)kv_cand_bytes / 1e6,
               100.0 * (double)kv_cand_bytes / (double)kv_bf16_bytes, slot_bf16 * 32.0 / 1e6,
               slot_cand * 32.0 / 1e6, 100.0 * slot_cand / slot_bf16);
        printf("PPL: bf16 %.4f -> %s %.4f (delta %+.3f%%)\n", ref.ppl,
               Engine::kv_mode_name(cand_mode), cand.ppl, ppl_delta);
        printf("top-1 agreement: %d/%zu = %.3f%%\n", agree, N - 1, agree_pct);
        printf("KL(ref||candidate) over %d positions: mean %.6g, p99 %.6g, max %.6g; mean "
               "|dlogit| %.6g\n",
               nS, kl_mean, kl_p99, kl_max, dlogit);
        printf("modelled attention KV read/token @4K: %.1f MB bf16 -> %.1f MB\n",
               kv_tok_bytes(Engine::KvMode::Bf16) / 1e6, kv_tok_bytes(cand_mode) / 1e6);

        const bool gate_doc = agree_pct >= 99.5 && kl_mean <= 1e-3 && kl_p99 <= 5e-3;
        const bool gate_success = gate_doc && ppl_delta <= 1.0;
        printf("quality gate (docs/CHECKPOINTS.md 3.4: top-1 >= 99.5%%, mean KL <= 1e-3, "
               "p99 KL <= 5e-3): %s\n", gate_doc ? "PASS" : "FAIL");
        printf("success criterion (PPL delta <= +1%% on top of the doc gate): %s\n",
               gate_success ? "PASS" : "FAIL");

        if (do_bench) {
            std::vector<int> body = ids;
            if (!body.empty() && body[0] == 124894) body.erase(body.begin());
            std::vector<int> ids4k;
            while ((int)ids4k.size() < ctx && !body.empty())
                ids4k.insert(ids4k.end(), body.begin(), body.end());
            if ((int)ids4k.size() > ctx) ids4k.resize(ctx);
            ids4k.insert(ids4k.begin(), 124894);
            if ((int)ids4k.size() > ctx) ids4k.resize(ctx);
            auto bench = [&](Engine::KvMode m, double& pre_s, double& dec_s) -> bool {
                if (!e.set_kv_mode(m, err)) { fprintf(stderr, "%s\n", err.c_str()); return false; }
                GenConfig cfg;
                cfg.max_tokens = decode_len;
                cfg.temperature = 0.0f;
                // Warm up pipeline compilation / first touch.
                e.reset();
                std::vector<int> warm(ids4k.begin(), ids4k.begin() + std::min<size_t>(128, ids4k.size()));
                e.prefill_ids(warm, true);
                int tok = e.sample(cfg);
                e.forward(tok, e.pos(), true);
                e.reset();
                auto t0 = std::chrono::steady_clock::now();
                e.prefill_ids(ids4k, true);
                auto t1 = std::chrono::steady_clock::now();
                tok = e.sample(cfg);
                auto t2 = std::chrono::steady_clock::now();
                for (int k = 1; k < decode_len; ++k) {
                    e.forward(tok, e.pos(), true);
                    tok = e.sample(cfg);
                }
                auto t3 = std::chrono::steady_clock::now();
                pre_s = std::chrono::duration<double>(t1 - t0).count();
                dec_s = std::chrono::duration<double>(t3 - t2).count();
                return true;
            };
            double pre_b = 0, dec_b = 0, pre_i = 0, dec_i = 0;
            if (!bench(Engine::KvMode::Bf16, pre_b, dec_b)) return 1;
            if (!bench(cand_mode, pre_i, dec_i)) return 1;
            const double dn = std::max(1, decode_len - 1);
            printf("\n4K bench (prefill %zu tokens, decode %d tokens, greedy):\n",
                   ids4k.size(), decode_len);
            printf("  bf16: prefill %.3f s = %.1f tok/s, decode %.3f s = %.2f tok/s\n", pre_b,
                   ids4k.size() / pre_b, dec_b, dn / dec_b);
            printf("  %s: prefill %.3f s = %.1f tok/s, decode %.3f s = %.2f tok/s\n",
                   Engine::kv_mode_name(cand_mode), pre_i, ids4k.size() / pre_i, dec_i,
                   dn / dec_i);
            printf("  decode speedup: %.3fx\n", dec_b / dec_i);
        }
    } else if (cmd == "ppl") {
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::string file = "docs/README.md";
        int max_tokens = 256;
        bool kv_i8 = false;
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--file") file = argv[i + 1];
            if (std::string(argv[i]) == "--tokens") max_tokens = atoi(argv[i + 1]);
        }
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--kv-int8") kv_i8 = true;
        if (kv_i8 && !e.set_kv_int8(true, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
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
        printf("ppl: %.3f over %zu tokens (%s, kv=%s)\n",
               std::exp(nll / (double)(ids.size() - 1)), ids.size() - 1, file.c_str(),
               Engine::kv_mode_name(e.kv_mode()));
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
    } else if (cmd == "test-mpp-tile") {
        // Does the tensor-core GEMM result per row depend on the M tile shape?
        // The sequential prefill uses gemm_mpp32 for T <= 32, the packed prefill
        // uses the 64-row gemm_mpp, so this gates bit-exactness of the two paths.
        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
        const Tensor& qw = w.get("language_model.model.layers.0.conv.in_proj.weight");
        const uint32_t N = (uint32_t)qw.rows(), K = (uint32_t)(qw.cols() * 8);
        const uint32_t off = e.weight_offset(qw);
        const uint32_t M = 23, rows = 64;
        MTL::Buffer* A = m.new_buffer((size_t)rows * K * 2);
        MTL::Buffer* C32 = m.new_buffer((size_t)rows * N * 4);
        MTL::Buffer* C64 = m.new_buffer((size_t)rows * N * 4);
        MTL::Buffer* C64b = m.new_buffer((size_t)rows * N * 4);
        fill_bf16(A, (size_t)rows * K, xfn);
        {
            auto cb = m.begin();
            auto enc = cb->computeCommandEncoder();
            gemm_mpp32(m, enc, A, e.bf16_buffer(), C32, M, N, K, off);
            gemm_mpp(m, enc, A, e.bf16_buffer(), C64, M, N, K, off);
            gemm_mpp(m, enc, A, e.bf16_buffer(), C64b, rows, N, K, off);
            enc->endEncoding();
            m.submit(cb, true);
        }
        const float* p32 = (const float*)C32->contents();
        const float* p64 = (const float*)C64->contents();
        const float* p64b = (const float*)C64b->contents();
        auto cmp = [&](const float* a, const float* b, const char* label) {
            double mx = 0.0;
            size_t ndiff = 0;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t c = 0; c < N; ++c) {
                    const float d = fabsf(a[(size_t)r * N + c] - b[(size_t)r * N + c]);
                    if (d != 0.0f) ++ndiff;
                    if (d > mx) mx = d;
                }
            printf("%-42s %zu/%zu differ, max abs diff %.6g\n", label, ndiff, (size_t)M * N, mx);
        };
        printf("test-mpp-tile: N=%u K=%u, rows 0..%u of 64\n", N, K, M - 1);
        cmp(p32, p64, "gemm_mpp32(M=23) vs gemm_mpp(M=23)");
        cmp(p64, p64b, "gemm_mpp(M=23) vs gemm_mpp(M=64)");
    } else if (cmd == "batch-sparse-test") {
        return cmd_batch_sparse(m, w);
    } else if (cmd == "batch-prefill-test") {
        // Packed multi-request prefill vs the sequential Engine path.
        Tokenizer t;
        if (!t.load("models/tokenizer.json", err)) {
            fprintf(stderr, "tokenizer: %s\n", err.c_str());
            return 1;
        }
        int nreq = 64, reps = 3, state_check = 4, long_target = 423;
        bool words = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--n" && i + 1 < argc) nreq = atoi(argv[++i]);
            else if (a == "--repeat" && i + 1 < argc) reps = atoi(argv[++i]);
            else if (a == "--state-check" && i + 1 < argc) state_check = atoi(argv[++i]);
            else if (a == "--long" && i + 1 < argc) long_target = atoi(argv[++i]);
            else if (a == "--words") words = true;
        }
        if (nreq < 1) nreq = 1;
        if (reps < 1) reps = 1;
        // Keep every packed request's first-token logits stable across passes so
        // they can be diffed against the sequential reference below.
        setenv("MIORI_BATCH_KEEP_LOGITS", "1", 1);

        Engine e(m, w);
        if (!e.init(err)) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
        BatchEngine be(m, w, e);
        if (!be.init(err)) { fprintf(stderr, "batch: %s\n", err.c_str()); return 1; }

        const std::string sys = "You are a helpful assistant.";
        // Default: single-word user turns, i.e. ~23-token prompts through the LFM3
        // chat template (the 64-concurrent server scenario). --words uses longer
        // (~29-token) question prompts instead.
        static const char* kWords[] = {"Hi",     "OK",   "Yes",  "No",   "Hey",  "Hello",
                                       "Thanks", "Sure", "Go",   "Stop", "Blue", "Red",
                                       "Cat",    "Dog",  "Sun",  "Moon", "Fast", "Slow",
                                       "One",    "Two",  "Nice", "Cool", "Wow",  "True"};
        const char* qs[4] = {"What is %d plus %d?", "Name color number %d.",
                             "Say hello to friend %d.", "What is %d times 2?"};
        std::vector<std::vector<int>> ids(nreq);
        size_t minlen = (size_t)-1, maxlen = 0, sumlen = 0;
        for (int i = 0; i < nreq; ++i) {
            if (words) {
                char buf[96];
                snprintf(buf, sizeof buf, qs[i % 4], i + 1, i + 2);
                ids[i] = t.chat(sys, buf, false);
            } else {
                ids[i] = t.chat(sys, kWords[i % (int)(sizeof(kWords) / sizeof(kWords[0]))],
                                false);
            }
            minlen = std::min(minlen, ids[i].size());
            maxlen = std::max(maxlen, ids[i].size());
            sumlen += ids[i].size();
        }

        GenConfig cfg;
        cfg.max_tokens = 1;
        cfg.temperature = 0.0f;
        const int waves = (nreq + BatchEngine::kBmax - 1) / BatchEngine::kBmax;

        // Warm up both paths (pipeline compilation / first-touch).
        {
            be.clear_slots();
            std::vector<std::vector<int>> warm(ids.begin(),
                                               ids.begin() + std::min<size_t>(4, ids.size()));
            be.add_batch(warm, 1);
            e.reset();
            e.prefill_ids(ids[0], true);
            e.sample(cfg);
        }

        std::vector<int> ref_first(nreq), pack_first(nreq, -1);
        std::vector<float> ref_logits((size_t)nreq * BatchEngine::kVocab);
        std::vector<float> pack_logits((size_t)nreq * BatchEngine::kVocab);
        std::vector<double> ref_time(reps), pack_time(reps);

        for (int rep = 0; rep < reps; ++rep) {
            const bool capture = (rep == 0);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < nreq; ++i) {
                e.reset();
                e.prefill_ids(ids[i], true);
                ref_first[i] = e.sample(cfg);
                if (capture)
                    memcpy(&ref_logits[(size_t)i * BatchEngine::kVocab], e.logits(),
                           (size_t)BatchEngine::kVocab * 4);
            }
            ref_time[rep] = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();

            double ps = 0.0;
            for (int w = 0; w < waves; ++w) {
                const int lo = w * BatchEngine::kBmax;
                const int hi = std::min(nreq, lo + BatchEngine::kBmax);
                std::vector<std::vector<int>> group(ids.begin() + lo, ids.begin() + hi);
                be.clear_slots();
                auto slots = be.add_batch(group, 1);
                ps += be.last_prefill_s();
                for (int k = 0; k < hi - lo; ++k) {
                    const int i = lo + k;
                    if (slots[k] >= 0 && !be.generated(slots[k]).empty())
                        pack_first[i] = be.generated(slots[k])[0];
                    if (capture && be.packed_logits(k))
                        memcpy(&pack_logits[(size_t)i * BatchEngine::kVocab],
                               be.packed_logits(k), (size_t)BatchEngine::kVocab * 4);
                }
            }
            pack_time[rep] = ps;
        }

        int idmatch = 0;
        int first_bad = -1;
        double maxdiff = 0.0;
        size_t logit_ndiff = 0;
        for (int i = 0; i < nreq; ++i) {
            if (pack_first[i] == ref_first[i]) ++idmatch;
            else if (first_bad < 0) first_bad = i;
            for (int v = 0; v < BatchEngine::kVocab; ++v) {
                const float d = fabsf(pack_logits[(size_t)i * BatchEngine::kVocab + v] -
                                      ref_logits[(size_t)i * BatchEngine::kVocab + v]);
                if (d != 0.0f) ++logit_ndiff;
                if (d > maxdiff) maxdiff = d;
            }
        }

        const double ref_med = median(ref_time), pack_med = median(pack_time);
        printf("\nbatch-prefill-test: %d prompts, %zu..%zu tokens (mean %.1f, total %zu), "
               "%d wave(s) of <= %d, %d repetition(s)\n",
               nreq, minlen, maxlen, (double)sumlen / nreq, sumlen, waves, BatchEngine::kBmax,
               reps);
        printf("sequential Engine::prefill_ids : %.3f s median (min %.3f max %.3f) = %.0f tok/s\n",
               ref_med, *std::min_element(ref_time.begin(), ref_time.end()),
               *std::max_element(ref_time.begin(), ref_time.end()), sumlen / ref_med);
        printf("packed BatchEngine::add_batch  : %.3f s median (min %.3f max %.3f) = %.0f tok/s\n",
               pack_med, *std::min_element(pack_time.begin(), pack_time.end()),
               *std::max_element(pack_time.begin(), pack_time.end()), sumlen / pack_med);
        printf("prefill speedup: %.2fx (%.2f -> %.2f ms per request)\n", ref_med / pack_med,
               1000.0 * ref_med / nreq, 1000.0 * pack_med / nreq);
        printf("first tokens: %d/%d identical%s\n", idmatch, nreq,
               first_bad >= 0 ? " (first mismatch at prompt index shown below)" : "");
        if (first_bad >= 0)
            printf("  first mismatch: prompt %d, ref=%d packed=%d\n", first_bad,
                   ref_first[first_bad], pack_first[first_bad]);
        printf("first-token logits: %zu/%zu elements differ, max |diff| = %.6g\n", logit_ndiff,
               (size_t)nreq * BatchEngine::kVocab, maxdiff);

        // ---- byte-exact state check: KV cache and conv state of the slots must
        // match what the sequential Engine leaves behind for the same prompt.
        auto state_cmp = [&](const std::vector<std::vector<int>>& group,
                             const std::vector<int>& slots, const char* tag) {
            int ok = 0;
            for (size_t i = 0; i < group.size(); ++i) {
                e.reset();
                e.prefill_ids(group[i], false);
                const int pos = e.pos();
                const int slot = slots[i];
                size_t kv_bad = 0, kv_bytes = 0, cs_bad = 0, cs_bytes = 0;
                for (int l = 0; l < BatchEngine::kLayers; ++l) {
                    if (Engine::is_attention_layer(l)) {
                        for (int h = 0; h < BatchEngine::kKvHeads; ++h) {
                            const size_t head = (size_t)slot * BatchEngine::kKvHeads + h;
                            if (e.k_int8()) {
                                const uint8_t* ek = (const uint8_t*)e.kv_k8(l)->contents();
                                const uint8_t* eks = (const uint8_t*)e.kv_ks(l)->contents();
                                const uint8_t* sk =
                                    (const uint8_t*)be.slot_kv_k8(l)->contents();
                                const uint8_t* sks =
                                    (const uint8_t*)be.slot_kv_ks(l)->contents();
                                const size_t kb = (size_t)pos * BatchEngine::kHeadDim;
                                const size_t skb = (size_t)pos * 4 * 2;
                                const size_t ebase = (size_t)h * BatchEngine::kMaxSeq *
                                                     BatchEngine::kHeadDim;
                                const size_t sbase = head * BatchEngine::kMaxSeq *
                                                     BatchEngine::kHeadDim;
                                kv_bad += diff_bytes(ek + ebase, sk + sbase, kb);
                                kv_bytes += kb;
                                const size_t esbase =
                                    (size_t)h * BatchEngine::kMaxSeq * 4 * 2;
                                const size_t ssbase = head * BatchEngine::kMaxSeq * 4 * 2;
                                kv_bad += diff_bytes(eks + esbase, sks + ssbase, skb);
                                kv_bytes += skb;
                            } else {
                                const uint8_t* ek = (const uint8_t*)e.kv_k(l)->contents();
                                const uint8_t* sk =
                                    (const uint8_t*)be.slot_kv_k(l)->contents();
                                const size_t kb = (size_t)pos * BatchEngine::kHeadDim * 2;
                                const size_t ebase = (size_t)h * BatchEngine::kMaxSeq *
                                                     BatchEngine::kHeadDim * 2;
                                const size_t sbase = head * BatchEngine::kMaxSeq *
                                                     BatchEngine::kHeadDim * 2;
                                kv_bad += diff_bytes(ek + ebase, sk + sbase, kb);
                                kv_bytes += kb;
                            }
                            if (e.v_int8()) {
                                const uint8_t* ev = (const uint8_t*)e.kv_v8(l)->contents();
                                const uint8_t* evs = (const uint8_t*)e.kv_vs(l)->contents();
                                const uint8_t* sv =
                                    (const uint8_t*)be.slot_kv_v8(l)->contents();
                                const uint8_t* svs =
                                    (const uint8_t*)be.slot_kv_vs(l)->contents();
                                const size_t vb = (size_t)pos;
                                const size_t sb = (size_t)pos * 2;
                                const size_t ebase_v = (size_t)h * BatchEngine::kHeadDim *
                                                       BatchEngine::kMaxSeq;
                                const size_t sbase_v = head * BatchEngine::kHeadDim *
                                                       BatchEngine::kMaxSeq;
                                for (int d = 0; d < BatchEngine::kHeadDim; ++d) {
                                    const size_t eo =
                                        ebase_v + (size_t)d * BatchEngine::kMaxSeq;
                                    const size_t so =
                                        sbase_v + (size_t)d * BatchEngine::kMaxSeq;
                                    kv_bad += diff_bytes(ev + eo, sv + so, vb);
                                    kv_bytes += vb;
                                }
                                for (int g = 0; g < 4; ++g) {
                                    const size_t eo =
                                        ((size_t)h * 4 + g) * BatchEngine::kMaxSeq * 2;
                                    const size_t so =
                                        (head * 4 + g) * BatchEngine::kMaxSeq * 2;
                                    kv_bad += diff_bytes(evs + eo, svs + so, sb);
                                    kv_bytes += sb;
                                }
                            } else {
                                const uint8_t* ev = (const uint8_t*)e.kv_v(l)->contents();
                                const uint8_t* sv =
                                    (const uint8_t*)be.slot_kv_v(l)->contents();
                                const size_t vb = (size_t)pos * 2;
                                const size_t ebase_v = (size_t)h * BatchEngine::kHeadDim *
                                                       BatchEngine::kMaxSeq * 2;
                                const size_t sbase_v = head * BatchEngine::kHeadDim *
                                                       BatchEngine::kMaxSeq * 2;
                                for (int d = 0; d < BatchEngine::kHeadDim; ++d) {
                                    const size_t eo = ebase_v + (size_t)d *
                                                       BatchEngine::kMaxSeq * 2;
                                    const size_t so = sbase_v + (size_t)d *
                                                       BatchEngine::kMaxSeq * 2;
                                    kv_bad += diff_bytes(ev + eo, sv + so, vb);
                                    kv_bytes += vb;
                                }
                            }
                        }
                    } else {
                        const uint8_t* ec = (const uint8_t*)e.conv_state(l)->contents();
                        const uint8_t* sc =
                            (const uint8_t*)be.slot_conv_state(l)->contents() +
                            ((size_t)slot * 2 + (size_t)be.conv_state_cur(l)) *
                                BatchEngine::kConvState * 4;
                        const size_t n = (size_t)BatchEngine::kConvState * 4;
                        cs_bad += diff_bytes(ec, sc, n);
                        cs_bytes += n;
                    }
                }
                printf("%s prompt %zu: %zu tokens, KV %zu/%zu bytes differ, conv %zu/%zu bytes "
                       "differ\n",
                       tag, i, group[i].size(), kv_bad, kv_bytes, cs_bad, cs_bytes);
                if (kv_bad == 0 && cs_bad == 0) ++ok;
            }
            return ok;
        };

        int state_ok = 0, state_total = 0;
        if (state_check > 0) {
            const int nc = std::min(state_check, nreq);
            std::vector<std::vector<int>> group(ids.begin(), ids.begin() + nc);
            be.clear_slots();
            auto slots = be.add_batch(group, 1);
            printf("\nstate check (packed KV/conv state vs sequential Engine):\n");
            state_ok += state_cmp(group, slots, "short");
            state_total += nc;
        }

        // ---- one long prompt (~423 tokens, several 128-token chunks), packed
        // together with a short prompt to exercise different request lengths.
        if (long_target > 0) {
            const std::string sentence =
                "This paragraph exists only to make the prompt long enough to span several "
                "prefill chunks. ";
            std::vector<int> long_ids;
            std::string text;
            do {
                text += sentence;
                long_ids = t.chat(sys, text, false);
            } while ((int)long_ids.size() < long_target);
            std::vector<std::vector<int>> group = {long_ids, ids[0]};

            e.reset();
            e.prefill_ids(group[0], true);
            const int ref_first_long = e.sample(cfg);
            std::vector<float> ref_logits_long(e.logits(), e.logits() + BatchEngine::kVocab);
            e.reset();
            e.prefill_ids(group[1], true);
            const int ref_first_short = e.sample(cfg);
            std::vector<float> ref_logits_short(e.logits(), e.logits() + BatchEngine::kVocab);

            be.clear_slots();
            auto slots = be.add_batch(group, 1);
            const double long_s = be.last_prefill_s();
            int long_ok = 0;
            int long_compared = 0;
            double long_max = 0.0;
            size_t long_ndiff = 0;
            for (int k = 0; k < 2; ++k) {
                const int pf = slots[k] >= 0 && !be.generated(slots[k]).empty()
                                   ? be.generated(slots[k])[0]
                                   : -1;
                const std::vector<float>& rl = k == 0 ? ref_logits_long : ref_logits_short;
                const int rf = k == 0 ? ref_first_long : ref_first_short;
                if (pf == rf) ++long_ok;
                if (!be.was_packed(k)) {
                    printf("  request %d (%zu tokens) used the sequential fallback "
                           "(prompt > kPackRows=%d); logits comparison skipped\n",
                           k, group[k].size(), BatchEngine::kPackRows);
                    continue;
                }
                ++long_compared;
                for (int v = 0; v < BatchEngine::kVocab; ++v) {
                    const float d = fabsf(be.packed_logits(k)[v] - rl[v]);
                    if (d != 0.0f) ++long_ndiff;
                    if (d > long_max) long_max = d;
                }
            }
            printf("\nlong prompt: %zu tokens + %zu-token prompt in one add_batch call: %.1f ms\n",
                   group[0].size(), group[1].size(), 1000.0 * long_s);
            printf("  first tokens: %d/2 identical, logits %zu/%zu differ (%d packed request(s)), "
                   "max |diff| %.6g\n",
                   long_ok, long_ndiff, (size_t)long_compared * BatchEngine::kVocab,
                   long_compared, long_max);
            printf("  state check:\n");
            state_ok += state_cmp(group, slots, "long+short");
            state_total += 2;
        }

        printf("\nstate check: %d/%d requests byte-identical (KV + conv state)\n", state_ok,
               state_total);
        const bool all_ok = idmatch == nreq && state_ok == state_total && logit_ndiff == 0;
        if (all_ok)
            printf("RESULT: packed prefill is bit-identical to the sequential path "
                   "(first-token logits, KV, conv state) and %.2fx faster\n",
                   ref_med / pack_med);
        else
            printf("RESULT: packed prefill differs from the sequential path "
                   "(ids %d/%d, logits %zu differ, state %d/%d); see above\n",
                   idmatch, nreq, logit_ndiff, state_ok, state_total);
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
