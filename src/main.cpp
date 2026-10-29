#include "metal.h"
#include "weights.h"
#include "ops.h"
#include "cpu_ref.h"
#include "tokenizer.h"
#include "engine.h"
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
    } else if (cmd == "serve") {
        int port = 8080;
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == "--port") port = atoi(argv[i + 1]);
        return run_server(m, w, port);
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
