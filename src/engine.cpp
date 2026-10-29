#include "engine.h"
#include "ops.h"
#include "cpu_ref.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include <stdexcept>

namespace miori {

static const char* kLayerSuffix[24] = {
    "conv", "conv", "self_attn", "conv", "self_attn", "conv", "self_attn", "conv",
    "self_attn", "conv", "self_attn", "conv", "self_attn", "conv", "self_attn", "conv",
    "self_attn", "conv", "self_attn", "conv", "self_attn", "conv", "self_attn", "conv",
};

bool Engine::init(std::string& err) {
    try {
        auto buf = [&](size_t bytes) { return m_.new_buffer(bytes); };
        x_ = buf(2048 * 4);
        h_ = buf(2048 * 2);
        emb_ = buf(2048 * 2);
        q_ = buf(2048 * 4);
        k_ = buf(512 * 4);
        v_ = buf(512 * 4);
        gate_ = buf(16 * 4);
        qn_ = buf(2048 * 2);
        kn_ = buf(512 * 2);
        attn_ = buf(2048 * 4);
        agated_ = buf(2048 * 2);
        proj_ = buf(2048 * 4);
        norm_ = buf(2048 * 4);
        fg_ = buf(7680 * 4);
        fu_ = buf(7680 * 4);
        fact_ = buf(7680 * 2);
        down_ = buf(2048 * 4);
        bcx_ = buf(6144 * 4);
        gated_ = buf(2048 * 4);
        conv_y_ = buf(2048 * 4);
        logits_ = buf(128000 * 4);
        tokbuf_ = buf(16);
        px_ = buf(128ull * 2048 * 4);
        ph_ = buf(128ull * 2048 * 2);
        pq_ = buf(128ull * 2048 * 4);
        pk_ = buf(128ull * 512 * 4);
        pv_ = buf(128ull * 512 * 4);
        pgate_ = buf(128ull * 16 * 4);
        pqn_ = buf(128ull * 2048 * 2);
        pkn_ = buf(128ull * 512 * 2);
        pattn_ = buf(128ull * 2048 * 4);
        pagated_ = buf(128ull * 2048 * 2);
        pproj_ = buf(128ull * 2048 * 4);
        pnorm_ = buf(128ull * 2048 * 4);
        pfg_ = buf(128ull * 7680 * 4);
        pfu_ = buf(128ull * 7680 * 4);
        pfact_ = buf(128ull * 7680 * 2);
        pdown_ = buf(128ull * 2048 * 4);
        pbcx_ = buf(128ull * 6144 * 4);
        pgated_ = buf(128ull * 2048 * 4);
        pconv_y_ = buf(128ull * 2048 * 4);
        scores_ = buf(128ull * 4096 * 4);
        scores_bf_ = buf(128ull * 4096 * 2);
        scratch_ = buf(15360ull * 2048 * 2);
        hlast_ = buf(2048 * 2);
        ptokens_ = buf(128 * 4);
        memset(x_->contents(), 0, 2048 * 4);
        memset(gate_->contents(), 0, 16 * 4);

        for (int l = 0; l < 24; ++l) {
            if (kLayerSuffix[l][0] == 's') {
                const size_t kbytes = (size_t)4096 * 512 * 2;
                kc_[l] = buf(kbytes);
                vc_[l] = buf(kbytes);
                memset(kc_[l]->contents(), 0, kbytes);
                memset(vc_[l]->contents(), 0, kbytes);
            } else {
                for (int k = 0; k < 2; ++k) {
                    cs_[l][k] = buf((size_t)6 * 2048 * 4);
                    memset(cs_[l][k]->contents(), 0, (size_t)6 * 2048 * 4);
                }
                cs_cur_[l] = 0;
            }
            // validate all tensors used by this layer
            const std::string base = "language_model.model.layers." + std::to_string(l) + ".";
            const char* names[] = {"operator_norm.weight", "post_operator_norm.weight",
                                   "ffn_norm.weight", "post_ffn_norm.weight"};
            for (const char* n : names) w_.get(base + n);
            if (kLayerSuffix[l][0] == 's') {
                for (const char* p : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                      "self_attn.o_proj", "self_attn.gate_proj"}) {
                    w_.get(base + p + ".weight");
                    w_.get(base + p + ".scales");
                    w_.get(base + p + ".biases");
                }
                w_.get(base + "self_attn.q_norm.weight");
                w_.get(base + "self_attn.k_norm.weight");
            } else {
                for (const char* p : {"conv.in_proj", "conv.out_proj"}) {
                    w_.get(base + p + ".weight");
                    w_.get(base + p + ".scales");
                    w_.get(base + p + ".biases");
                }
                w_.get(base + "conv.conv.weight");
            }
            for (const char* p : {"feed_forward.w1", "feed_forward.w2", "feed_forward.w3"}) {
                w_.get(base + p + ".weight");
                w_.get(base + p + ".scales");
                w_.get(base + p + ".biases");
            }
        }
        w_.get("language_model.model.embedding_norm.weight");
        w_.get("language_model.model.embed_tokens.weight");
        w_.get("language_model.model.embed_tokens.scales");
        wbf_ = buf(4200ull * 1024 * 1024);
        cb_ = m_.begin();
        e_ = cb_->computeCommandEncoder();
        auto reg = [&](const Tensor& q, const Tensor& s, const Tensor& b) {
            const uint32_t N = (uint32_t)q.rows();
            const uint32_t K = (uint32_t)(q.cols() * 8);
            uint32_t off = (wbf_used_ + 255u) & ~255u;
            woff_[q.name] = off;
            if (N % 64u == 0u) {
                dequant_tile(m_, e_, w_, q, s, b, wbf_, N, K, off);
            } else {
                dequant4(m_, e_, w_, q, s, b, wbf_, N, K, off);
            }
            wbf_used_ = off + (uint32_t)((uint64_t)N * K * 2ull);
        };
        for (int l = 0; l < 24; ++l) {
            const char* names[5];
            int nn = 0;
            if (kLayerSuffix[l][0] == 's') {
                names[nn++] = "self_attn.q_proj";
                names[nn++] = "self_attn.k_proj";
                names[nn++] = "self_attn.v_proj";
                names[nn++] = "self_attn.gate_proj";
                names[nn++] = "self_attn.o_proj";
            } else {
                names[nn++] = "conv.in_proj";
                names[nn++] = "conv.out_proj";
            }
            for (int i = 0; i < nn; ++i) {
                const Tensor *q, *s, *b;
                uint32_t N, K;
                qw(l, names[i], q, s, b, N, K);
                reg(*q, *s, *b);
            }
            for (const char* p : {"feed_forward.w1", "feed_forward.w2", "feed_forward.w3"}) {
                const Tensor *q, *s, *b;
                uint32_t N, K;
                qw(l, p, q, s, b, N, K);
                reg(*q, *s, *b);
            }
        }
        e_->endEncoding();
        m_.submit(cb_, true);
        printf("engine: bf16 weight cache %.2f GB\n", wbf_used_ / 1e9);
        printf("engine: buffers ready, KV %.1f MB\n", 11.0 * 4096 * 512 * 2 * 2 / 1e6);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

const Tensor& Engine::norm(int layer, const char* which) const {
    return w_.get("language_model.model.layers." + std::to_string(layer) + "." + which);
}

void Engine::qw(int layer, const char* proj, const Tensor*& q, const Tensor*& s,
                const Tensor*& b, uint32_t& N, uint32_t& K) const {
    const std::string base = "language_model.model.layers." + std::to_string(layer) + "." + proj;
    q = &w_.get(base + ".weight");
    s = &w_.get(base + ".scales");
    b = &w_.get(base + ".biases");
    N = (uint32_t)q->rows();
    K = (uint32_t)(q->cols() * 8);
}

void Engine::post_norm_add(int layer, const char* norm_name, MTL::Buffer* src) {
    postnorm_add(m_, e_, x_, src, w_, norm(layer, norm_name), x_, 1, 2048, 1e-5f);
}

void Engine::ffn(int layer) {
    rmsnorm(m_, e_, x_, w_, norm(layer, "ffn_norm.weight"), h_, 1, 2048, 1e-5f);
    const Tensor *q1, *s1, *b1, *q3, *s3, *b3, *q2, *s2, *b2;
    uint32_t N1, K1, N3, K3, N2, K2;
    qw(layer, "feed_forward.w1", q1, s1, b1, N1, K1);
    qw(layer, "feed_forward.w3", q3, s3, b3, N3, K3);
    gemv4(m_, e_, w_, *q1, *s1, *b1, h_, fg_, N1, K1);
    gemv4(m_, e_, w_, *q3, *s3, *b3, h_, fu_, N3, K3);
    silu_mul(m_, e_, fg_, fu_, fact_, 7680);
    qw(layer, "feed_forward.w2", q2, s2, b2, N2, K2);
    gemv4(m_, e_, w_, *q2, *s2, *b2, fact_, down_, N2, K2);
    post_norm_add(layer, "post_ffn_norm.weight", down_);
}

void Engine::attn_layer(int layer, int pos) {
    rmsnorm(m_, e_, x_, w_, norm(layer, "operator_norm.weight"), h_, 1, 2048, 1e-5f);
    const Tensor *qq, *qs, *qb, *kq, *ks, *kb, *vq, *vs, *vb, *gq, *gs, *gb;
    uint32_t qn, qk, kn, kk, vn, vk, gn, gk;
    qw(layer, "self_attn.q_proj", qq, qs, qb, qn, qk);
    qw(layer, "self_attn.k_proj", kq, ks, kb, kn, kk);
    qw(layer, "self_attn.v_proj", vq, vs, vb, vn, vk);
    qw(layer, "self_attn.gate_proj", gq, gs, gb, gn, gk);
    gemv4(m_, e_, w_, *qq, *qs, *qb, h_, q_, qn, qk);
    gemv4(m_, e_, w_, *kq, *ks, *kb, h_, k_, kn, kk);
    gemv4(m_, e_, w_, *vq, *vs, *vb, h_, v_, vn, vk);
    gemv4(m_, e_, w_, *gq, *gs, *gb, h_, gate_, gn, gk);
    // full/sliding decided by config order below
    const int type = (layer == 2 || layer == 6 || layer == 10 || layer == 14 ||
                      layer == 18 || layer == 22) ? 1 : 2;
    const bool partial = (type == 1);
    const float theta = partial ? 1000000.0f : 10000.0f;
    rmsnorm(m_, e_, q_, w_, w_.get("language_model.model.layers." + std::to_string(layer) +
                                   ".self_attn.q_norm.weight"),
            qn_, 16, 128, 1e-5f);
    rmsnorm(m_, e_, k_, w_, w_.get("language_model.model.layers." + std::to_string(layer) +
                                   ".self_attn.k_norm.weight"),
            kn_, 4, 128, 1e-5f);
    rope(m_, e_, qn_, kn_, (uint32_t)pos, theta, partial);
    kv_append(m_, e_, kn_, v_, kc_[layer], vc_[layer], (uint32_t)pos, 4096, 4);
    attn_decode(m_, e_, qn_, kc_[layer], vc_[layer], attn_, (uint32_t)pos, 4096, 4,
                (type == 2) ? 256u : 0u);
    gate_mul(m_, e_, attn_, gate_, agated_, 2048);
    const Tensor *oq, *os, *ob;
    uint32_t on, ok;
    qw(layer, "self_attn.o_proj", oq, os, ob, on, ok);
    gemv4(m_, e_, w_, *oq, *os, *ob, agated_, proj_, on, ok);
    post_norm_add(layer, "post_operator_norm.weight", proj_);
    ffn(layer);
}

void Engine::conv_layer(int layer, int pos) {
    rmsnorm(m_, e_, x_, w_, norm(layer, "operator_norm.weight"), h_, 1, 2048, 1e-5f);
    const Tensor *iq, *is, *ib, *oq, *os, *ob;
    uint32_t in, ik, on, ok;
    qw(layer, "conv.in_proj", iq, is, ib, in, ik);
    gemv4(m_, e_, w_, *iq, *is, *ib, h_, bcx_, in, ik);
    conv_gate(m_, e_, bcx_, gated_, 1, 2048);
    int cur = cs_cur_[layer];
    conv7(m_, e_, gated_, bcx_, w_, w_.get("language_model.model.layers." +
                                          std::to_string(layer) + ".conv.conv.weight"),
          cs_[layer][cur], cs_[layer][1 - cur], conv_y_, 1, 2048);
    cs_cur_[layer] = 1 - cur;
    to_bf16(m_, e_, conv_y_, h_, 2048);
    qw(layer, "conv.out_proj", oq, os, ob, on, ok);
    gemv4(m_, e_, w_, *oq, *os, *ob, h_, proj_, on, ok);
    post_norm_add(layer, "post_operator_norm.weight", proj_);
    ffn(layer);
}


void Engine::p_gemm(MTL::Buffer* A, uint32_t lda, const Tensor& q, const Tensor& s,
                    const Tensor& b, uint32_t N, uint32_t K, MTL::Buffer* C,
                    uint32_t ldc, uint32_t T) {
    const uint32_t off = woff_.at(q.name);
    if (N % 64u == 0u) {
        gemm_mpp(m_, e_, A, wbf_, C, T, N, K, off);
    } else {
        gemm_naive(m_, e_, A, wbf_, C, T, N, K, lda, K, ldc, 0, off, 0);
    }
}

void Engine::p_post(int layer, const char* norm_name, MTL::Buffer* src, int T) {
    postnorm_add(m_, e_, px_, src, w_, norm(layer, norm_name), px_, (uint32_t)T, 2048, 1e-5f);
}

void Engine::p_ffn(int layer, int T) {
    rmsnorm(m_, e_, px_, w_, norm(layer, "ffn_norm.weight"), ph_, T, 2048, 1e-5f);
    const Tensor *q1, *s1, *b1, *q3, *s3, *b3, *q2, *s2, *b2;
    uint32_t N1, K1, N3, K3, N2, K2;
    qw(layer, "feed_forward.w1", q1, s1, b1, N1, K1);
    qw(layer, "feed_forward.w3", q3, s3, b3, N3, K3);
    p_gemm(ph_, 2048, *q1, *s1, *b1, N1, K1, pfg_, 7680, T);
    p_gemm(ph_, 2048, *q3, *s3, *b3, N3, K3, pfu_, 7680, T);
    silu_mul(m_, e_, pfg_, pfu_, pfact_, (uint32_t)(T * 7680));
    qw(layer, "feed_forward.w2", q2, s2, b2, N2, K2);
    p_gemm(pfact_, 7680, *q2, *s2, *b2, N2, K2, pdown_, 2048, T);
    p_post(layer, "post_ffn_norm.weight", pdown_, T);
}

void Engine::p_attn(int layer, int T, int start) {
    rmsnorm(m_, e_, px_, w_, norm(layer, "operator_norm.weight"), ph_, T, 2048, 1e-5f);
    const Tensor *qq, *qs, *qb, *kq, *ks, *kb, *vq, *vs, *vb, *gq, *gs, *gb;
    uint32_t qn, qk, kn, kk, vn, vk, gn, gk;
    qw(layer, "self_attn.q_proj", qq, qs, qb, qn, qk);
    qw(layer, "self_attn.k_proj", kq, ks, kb, kn, kk);
    qw(layer, "self_attn.v_proj", vq, vs, vb, vn, vk);
    qw(layer, "self_attn.gate_proj", gq, gs, gb, gn, gk);
    p_gemm(ph_, 2048, *qq, *qs, *qb, qn, qk, pq_, 2048, T);
    p_gemm(ph_, 2048, *kq, *ks, *kb, kn, kk, pk_, 512, T);
    p_gemm(ph_, 2048, *vq, *vs, *vb, vn, vk, pv_, 512, T);
    p_gemm(ph_, 2048, *gq, *gs, *gb, gn, gk, pgate_, 16, T);
    rmsnorm(m_, e_, pq_, w_, w_.get("language_model.model.layers." + std::to_string(layer) +
                                   ".self_attn.q_norm.weight"),
            pqn_, T * 16, 128, 1e-5f);
    rmsnorm(m_, e_, pk_, w_, w_.get("language_model.model.layers." + std::to_string(layer) +
                                   ".self_attn.k_norm.weight"),
            pkn_, T * 4, 128, 1e-5f);
    const int type = (layer == 2 || layer == 6 || layer == 10 || layer == 14 ||
                      layer == 18 || layer == 22) ? 1 : 2;
    const bool partial = (type == 1);
    const float theta = partial ? 1000000.0f : 10000.0f;
    rope_b(m_, e_, pqn_, pkn_, T, (uint32_t)start, theta, partial);
    kv_write_b(m_, e_, pkn_, pv_, kc_[layer], vc_[layer], T, (uint32_t)start, 4096);
    const uint32_t seq = (uint32_t)(start + T);
    const uint32_t kbase = (uint32_t)4096 * 128 * 2;
    const uint32_t vbase = (uint32_t)128 * 4096 * 2;
    for (uint32_t h = 0; h < 16; ++h) {
        const uint32_t kvh = h / 4u;
        gemm_naive(m_, e_, pqn_, kc_[layer], scores_, T, seq, 128,
                   2048, 128, seq, h * 128 * 2, kvh * kbase, 0);
        softmax_prefill(m_, e_, scores_, scores_bf_, T, seq, (uint32_t)start,
                        type == 2);
        gemm_naive(m_, e_, scores_bf_, vc_[layer], pattn_, T, 128, seq,
                   seq, 4096, 2048, 0, kvh * vbase, h * 128 * 4);
    }
    gate_mul_b(m_, e_, pattn_, pgate_, pagated_, T * 2048);
    const Tensor *oq, *os, *ob;
    uint32_t on, ok;
    qw(layer, "self_attn.o_proj", oq, os, ob, on, ok);
    p_gemm(pagated_, 2048, *oq, *os, *ob, on, ok, pproj_, 2048, T);
    p_post(layer, "post_operator_norm.weight", pproj_, T);
    p_ffn(layer, T);
}

void Engine::p_conv(int layer, int T) {
    rmsnorm(m_, e_, px_, w_, norm(layer, "operator_norm.weight"), ph_, T, 2048, 1e-5f);
    const Tensor *iq, *is, *ib, *oq, *os, *ob;
    uint32_t in, ik, on, ok;
    qw(layer, "conv.in_proj", iq, is, ib, in, ik);
    p_gemm(ph_, 2048, *iq, *is, *ib, in, ik, pbcx_, 6144, T);
    conv_gate(m_, e_, pbcx_, pgated_, T, 2048);
    int cur = cs_cur_[layer];
    conv7(m_, e_, pgated_, pbcx_, w_, w_.get("language_model.model.layers." +
                                             std::to_string(layer) + ".conv.conv.weight"),
          cs_[layer][cur], cs_[layer][1 - cur], pconv_y_, T, 2048);
    cs_cur_[layer] = 1 - cur;
    to_bf16(m_, e_, pconv_y_, ph_, T * 2048);
    qw(layer, "conv.out_proj", oq, os, ob, on, ok);
    p_gemm(ph_, 2048, *oq, *os, *ob, on, ok, pproj_, 2048, T);
    p_post(layer, "post_operator_norm.weight", pproj_, T);
    p_ffn(layer, T);
}

void Engine::prefill_chunk(const int* toks, int T, int start, bool need_logits) {
    uint32_t* tp = (uint32_t*)ptokens_->contents();
    for (int i = 0; i < T; ++i) tp[i] = (uint32_t)toks[i];
    cb_ = m_.begin();
    e_ = cb_->computeCommandEncoder();
    const Tensor& eq = w_.get("language_model.model.embed_tokens.weight");
    const Tensor& es = w_.get("language_model.model.embed_tokens.scales");
    const Tensor& eb = w_.get("language_model.model.embed_tokens.biases");
    embed8(m_, e_, w_, eq, es, eb, ptokens_, ph_, T, 2048);
    to_f32(m_, e_, ph_, px_, T * 2048);
    for (int l = 0; l < 24; ++l) {
        if (kLayerSuffix[l][0] == 's') p_attn(l, T, start);
        else p_conv(l, T);
    }
    if (need_logits) {
        rmsnorm(m_, e_, px_, w_, w_.get("language_model.model.embedding_norm.weight"),
                ph_, T, 2048, 1e-5f);
        copy_row(m_, e_, ph_, hlast_, 2048, T - 1, 2048);
        gemv8(m_, e_, w_, eq, es, eb, hlast_, logits_, 128000, 2048);
    }
    e_->endEncoding();
    m_.submit(cb_, true);
    pos_ = start + T;
}
void Engine::forward(int token, int pos, bool need_logits) {
    ((uint32_t*)tokbuf_->contents())[0] = (uint32_t)token;
    cb_ = m_.begin();
    e_ = cb_->computeCommandEncoder();
    const Tensor& eq = w_.get("language_model.model.embed_tokens.weight");
    const Tensor& es = w_.get("language_model.model.embed_tokens.scales");
    const Tensor& eb = w_.get("language_model.model.embed_tokens.biases");
    embed8(m_, e_, w_, eq, es, eb, tokbuf_, emb_, 1, 2048);
    to_f32(m_, e_, emb_, x_, 2048);
    for (int l = 0; l < 24; ++l) {
        if (kLayerSuffix[l][0] == 's') attn_layer(l, pos);
        else conv_layer(l, pos);
    }
    if (need_logits) {
        rmsnorm(m_, e_, x_, w_, w_.get("language_model.model.embedding_norm.weight"),
                h_, 1, 2048, 1e-5f);
        gemv8(m_, e_, w_, eq, es, eb, h_, logits_, 128000, 2048);
    }
    e_->endEncoding();
    m_.submit(cb_, true);
    pos_ = pos + 1;
}

int Engine::sample(const GenConfig& cfg) {
    const float* lg = (const float*)logits_->contents();
    if (cfg.temperature <= 0.0f) {
        int best = 0;
        float bv = lg[0];
        for (int i = 1; i < 128000; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        return best;
    }
    std::vector<int> idx(128000);
    for (int i = 0; i < 128000; ++i) idx[i] = i;
    const int k = std::max(1, std::min(cfg.top_k, 128000));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize(k);
    float mx = lg[idx[0]];
    std::vector<float> p(k);
    float sum = 0;
    for (int i = 0; i < k; ++i) { p[i] = std::exp((lg[idx[i]] - mx) / cfg.temperature); sum += p[i]; }
    for (float& x : p) x /= sum;
    if (cfg.top_p < 1.0f) {
        float c = 0;
        int n = 0;
        for (; n < k; ++n) { c += p[n]; if (c >= cfg.top_p) { ++n; break; } }
        p.resize(n); idx.resize(n);
        sum = 0; for (float x : p) sum += x;
        for (float& x : p) x /= sum;
    }
    static std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<float> dist(0, 1);
    float r = dist(rng), c = 0;
    for (size_t i = 0; i < p.size(); ++i) { c += p[i]; if (r <= c) return idx[i]; }
    return idx.back();
}

void Engine::reset() {
    pos_ = 0;
    for (int l = 0; l < 24; ++l) {
        if (kLayerSuffix[l][0] == 's') {
            memset(kc_[l]->contents(), 0, (size_t)4096 * 512 * 2);
            memset(vc_[l]->contents(), 0, (size_t)4096 * 512 * 2);
        } else {
            for (int k = 0; k < 2; ++k)
                memset(cs_[l][k]->contents(), 0, (size_t)6 * 2048 * 4);
            cs_cur_[l] = 0;
        }
    }
}

std::vector<int> Engine::generate(const std::vector<int>& prompt, const GenConfig& cfg) {
    reset();
    std::vector<int> out;
    auto t0 = std::chrono::steady_clock::now();
    size_t p = 0;
    if (cfg.slow_prefill) {
        for (size_t i = 0; i < prompt.size(); ++i) forward(prompt[i], (int)i, i + 1 == prompt.size());
    } else while (p < prompt.size()) {
        int n = (int)std::min<size_t>(128, prompt.size() - p);
        prefill_chunk(prompt.data() + p, n, (int)p, p + n == prompt.size());
        p += (size_t)n;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (prompt.empty()) return out;
    int tok = sample(cfg);
    out.push_back(tok);
    for (int t = 1; t < cfg.max_tokens; ++t) {
        if (tok == 124900) break;
        forward(tok, pos_, true);
        tok = sample(cfg);
        out.push_back(tok);
    }
    auto t2 = std::chrono::steady_clock::now();
    double psec = std::chrono::duration<double>(t1 - t0).count();
    double dsec = std::chrono::duration<double>(t2 - t1).count();
    last_prefill_s_ = psec;
    last_decode_s_ = dsec;
    fprintf(stderr, "[prefill %zu tok %.3fs %.1f tok/s | decode %zu tok %.3fs %.1f tok/s]\n",
            prompt.size(), psec, prompt.size() / (psec + 1e-9), out.size(), dsec,
            out.size() / (dsec + 1e-9));
    return out;
}

const float* Engine::logits() const { return (const float*)logits_->contents(); }

}  // namespace miori
