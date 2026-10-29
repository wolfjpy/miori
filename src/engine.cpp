#include "engine.h"
#include "ops.h"
#include "cpu_ref.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
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
        pacc_ = buf(4ull * 16 * 4 * 128 * 4);
        pm_ = buf(4ull * 16 * 4 * 4);
        pl_ = buf(4ull * 16 * 4 * 4);
        ptokens_ = buf(128 * 4);
        memset(x_->contents(), 0, 2048 * 4);
        memset(gate_->contents(), 0, 16 * 4);

        for (int l = 0; l < 24; ++l) {
            if (kLayerSuffix[l][0] != 's') {
                for (int k = 0; k < 2; ++k) {
                    cs_[l][k] = buf((size_t)6 * 2048 * 4);
                    memset(cs_[l][k]->contents(), 0, (size_t)6 * 2048 * 4);
                }
                cs_cur_[l] = 0;
                for (int s = 0; s < kSnapshotSlots; ++s)
                    snap_cs_[s][l] = buf((size_t)6 * 2048 * 4);
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
        // KV format: bf16 by default; MIORI_KV_INT8=1 selects full int8 and
        // MIORI_KV_MIX=kv|vk selects a mixed K/V cache (the letter is the
        // quantised side). Only the chosen side formats are allocated.
        KvMode want = KvMode::Bf16;
        if (const char* mix = getenv("MIORI_KV_MIX")) {
            if (std::strcmp(mix, "kv") == 0) want = KvMode::K8V16;
            else if (std::strcmp(mix, "vk") == 0) want = KvMode::K16V8;
            else {
                err = std::string("MIORI_KV_MIX must be kv or vk, got '") + mix + "'";
                return false;
            }
        } else if (const char* e = getenv("MIORI_KV_INT8"); e && atoi(e) != 0) {
            want = KvMode::Int8;
        }
        kv_mode_ = want;
        if (!alloc_kv_for_mode(want, err)) return false;
        dummy_bf16_ = buf(4096);
        dummy_i8_ = buf(4096);
        dummy_scale_ = buf(4096);
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
        printf("engine: buffers ready, KV %.1f MB (%s)\n", kv_cache_bytes() / 1e6,
               kv_mode_name(kv_mode_));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

// ---- KV allocation ---------------------------------------------------------
// One allocator per side/format, idempotent so set_kv_mode() can add the
// buffers a mixed mode needs without disturbing the ones already live.

bool Engine::alloc_k_bf16(std::string& err) {
    try {
        const size_t kbytes = (size_t)kMaxSeq * kKvHeads * kHeadDim * 2;
        for (int l = 0; l < 24; ++l) {
            if (!is_attention_layer(l) || kc_[l]) continue;
            kc_[l] = m_.new_buffer(kbytes);
            memset(kc_[l]->contents(), 0, kbytes);
            for (int s = 0; s < kSnapshotSlots; ++s)
                if (!snap_kc_[s][l]) snap_kc_[s][l] = m_.new_buffer(kbytes);
        }
    } catch (const std::exception& e) {
        err = std::string("kv bf16 K alloc: ") + e.what();
        return false;
    }
    return true;
}

bool Engine::alloc_v_bf16(std::string& err) {
    try {
        const size_t kbytes = (size_t)kMaxSeq * kKvHeads * kHeadDim * 2;
        for (int l = 0; l < 24; ++l) {
            if (!is_attention_layer(l) || vc_[l]) continue;
            vc_[l] = m_.new_buffer(kbytes);
            memset(vc_[l]->contents(), 0, kbytes);
            for (int s = 0; s < kSnapshotSlots; ++s)
                if (!snap_vc_[s][l]) snap_vc_[s][l] = m_.new_buffer(kbytes);
        }
    } catch (const std::exception& e) {
        err = std::string("kv bf16 V alloc: ") + e.what();
        return false;
    }
    return true;
}

bool Engine::alloc_k_int8(std::string& err) {
    try {
        const size_t k8 = (size_t)kKvHeads * kMaxSeq * kHeadDim;   // 2 MiB / layer
        // one bf16 scale per (head, token, 32-wide d group): [4][4096][4]
        const size_t s8 = (size_t)kKvHeads * kMaxSeq * 4 * 2;      // 128 KiB / layer
        for (int l = 0; l < 24; ++l) {
            if (!is_attention_layer(l) || kc8_[l]) continue;
            kc8_[l] = m_.new_buffer(k8);
            ks8_[l] = m_.new_buffer(s8);
            memset(kc8_[l]->contents(), 0, k8);
            memset(ks8_[l]->contents(), 0, s8);
            for (int s = 0; s < kSnapshotSlots; ++s) {
                if (!snap_kc8_[s][l]) {
                    snap_kc8_[s][l] = m_.new_buffer(k8);
                    snap_ks8_[s][l] = m_.new_buffer(s8);
                }
            }
        }
    } catch (const std::exception& e) {
        err = std::string("kv int8 K alloc: ") + e.what();
        return false;
    }
    return true;
}

bool Engine::alloc_v_int8(std::string& err) {
    try {
        const size_t v8 = (size_t)kKvHeads * kMaxSeq * kHeadDim;
        const size_t s8 = (size_t)kKvHeads * kMaxSeq * 4 * 2;
        for (int l = 0; l < 24; ++l) {
            if (!is_attention_layer(l) || vc8_[l]) continue;
            vc8_[l] = m_.new_buffer(v8);
            vs8_[l] = m_.new_buffer(s8);
            memset(vc8_[l]->contents(), 0, v8);
            memset(vs8_[l]->contents(), 0, s8);
            for (int s = 0; s < kSnapshotSlots; ++s) {
                if (!snap_vc8_[s][l]) {
                    snap_vc8_[s][l] = m_.new_buffer(v8);
                    snap_vs8_[s][l] = m_.new_buffer(s8);
                }
            }
        }
    } catch (const std::exception& e) {
        err = std::string("kv int8 V alloc: ") + e.what();
        return false;
    }
    return true;
}

bool Engine::alloc_kv_for_mode(KvMode m, std::string& err) {
    if (kv_mode_k_int8(m)) {
        if (!alloc_k_int8(err)) return false;
    } else if (!alloc_k_bf16(err)) {
        return false;
    }
    if (kv_mode_v_int8(m)) {
        if (!alloc_v_int8(err)) return false;
    } else if (!alloc_v_bf16(err)) {
        return false;
    }
    return true;
}

bool Engine::set_kv_mode(KvMode m, std::string& err) {
    if (!alloc_kv_for_mode(m, err)) return false;
    // Data is not migrated: the caller has to reset() and refill the cache.
    kv_mode_ = m;
    return true;
}

bool Engine::set_kv_int8(bool on, std::string& err) {
    return set_kv_mode(on ? KvMode::Int8 : KvMode::Bf16, err);
}

const char* Engine::kv_mode_name(KvMode m) {
    switch (m) {
        case KvMode::Bf16: return "bf16";
        case KvMode::Int8: return "int8";
        case KvMode::K8V16: return "K int8 / V bf16";
        case KvMode::K16V8: return "K bf16 / V int8";
    }
    return "?";
}

size_t Engine::kv_cache_bytes() const {
    size_t n = 0;
    for (int l = 0; l < 24; ++l) {
        if (!is_attention_layer(l)) continue;
        if (k_int8() && kc8_[l]) n += kc8_[l]->length() + ks8_[l]->length();
        else if (kc_[l]) n += kc_[l]->length();
        if (v_int8() && vc8_[l]) n += vc8_[l]->length() + vs8_[l]->length();
        else if (vc_[l]) n += vc_[l]->length();
    }
    return n;
}

MTL::Buffer* Engine::kv_k(int layer) const {
    return (!k_int8() && is_attention_layer(layer)) ? kc_[layer] : nullptr;
}

MTL::Buffer* Engine::kv_v(int layer) const {
    return (!v_int8() && is_attention_layer(layer)) ? vc_[layer] : nullptr;
}

MTL::Buffer* Engine::kv_k8(int layer) const {
    return (k_int8() && is_attention_layer(layer)) ? kc8_[layer] : nullptr;
}

MTL::Buffer* Engine::kv_v8(int layer) const {
    return (v_int8() && is_attention_layer(layer)) ? vc8_[layer] : nullptr;
}

MTL::Buffer* Engine::kv_ks(int layer) const {
    return (k_int8() && is_attention_layer(layer)) ? ks8_[layer] : nullptr;
}

MTL::Buffer* Engine::kv_vs(int layer) const {
    return (v_int8() && is_attention_layer(layer)) ? vs8_[layer] : nullptr;
}

bool Engine::is_attention_layer(int layer) {
    return layer >= 0 && layer < 24 && kLayerSuffix[layer][0] == 's';
}

int Engine::layer_kind(int layer) {
    if (!is_attention_layer(layer)) return 0;
    switch (layer) {
        case 2: case 6: case 10: case 14: case 18: case 22:
            return 1;   // full attention
        default:
            return 2;   // sliding-window attention
    }
}

uint32_t Engine::weight_offset(const Tensor& t) const {
    auto it = woff_.find(t.name);
    return it == woff_.end() ? 0u : it->second;
}

MTL::Buffer* Engine::conv_state(int layer) const {
    return is_attention_layer(layer) ? nullptr : cs_[layer][cs_cur_[layer]];
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
    const int type = layer_kind(layer);
    const bool partial = (type == 1);
    const float theta = partial ? 1000000.0f : 10000.0f;
    rmsnorm(m_, e_, q_, w_, w_.get("language_model.model.layers." + std::to_string(layer) +
                                   ".self_attn.q_norm.weight"),
            qn_, 16, 128, 1e-5f);
    rmsnorm(m_, e_, k_, w_, w_.get("language_model.model.layers." + std::to_string(layer) +
                                   ".self_attn.k_norm.weight"),
            kn_, 4, 128, 1e-5f);
    rope(m_, e_, qn_, kn_, (uint32_t)pos, theta, partial);
    const KvMode mode = kv_mode_;
    if (mode == KvMode::Int8) {
        kv_append_i8(m_, e_, kn_, v_, kc8_[layer], vc8_[layer], ks8_[layer], vs8_[layer],
                     (uint32_t)pos, kMaxSeq, 4);
    } else if (mode == KvMode::Bf16) {
        kv_append(m_, e_, kn_, v_, kc_[layer], vc_[layer], (uint32_t)pos, 4096, 4);
    } else if (k_int8()) {   // K int8 + V bf16
        kv_append_i8(m_, e_, kn_, v_, kc8_[layer], dummy_i8_, ks8_[layer], dummy_scale_,
                     (uint32_t)pos, kMaxSeq, 4, 1);
        kv_append(m_, e_, kn_, v_, dummy_bf16_, vc_[layer], (uint32_t)pos, 4096, 4, 2);
    } else {                 // K bf16 + V int8
        kv_append(m_, e_, kn_, v_, kc_[layer], dummy_bf16_, (uint32_t)pos, 4096, 4, 1);
        kv_append_i8(m_, e_, kn_, v_, dummy_i8_, vc8_[layer], dummy_scale_, vs8_[layer],
                     (uint32_t)pos, kMaxSeq, 4, 2);
    }
    const uint32_t window = (type == 2) ? 256u : 0u;
    if (mode == KvMode::Int8) {
        if (type == 1 && pos >= 512) {
            const uint32_t splits = 16;
            attn_decode_split_i8(m_, e_, qn_, kc8_[layer], vc8_[layer], ks8_[layer],
                                 vs8_[layer], pacc_, pm_, pl_, (uint32_t)pos, kMaxSeq, 4,
                                 splits);
            attn_combine(m_, e_, pacc_, pm_, pl_, attn_, splits);
        } else {
            attn_decode_i8(m_, e_, qn_, kc8_[layer], vc8_[layer], ks8_[layer], vs8_[layer],
                           attn_, (uint32_t)pos, kMaxSeq, 4, window);
        }
    } else if (mode == KvMode::Bf16) {
        if (type == 1 && pos >= 512) {
            const uint32_t splits = 16;
            attn_decode_split(m_, e_, qn_, kc_[layer], vc_[layer], pacc_, pm_, pl_,
                              (uint32_t)pos, 4096, 4, splits);
            attn_combine(m_, e_, pacc_, pm_, pl_, attn_, splits);
        } else {
            attn_decode(m_, e_, qn_, kc_[layer], vc_[layer], attn_, (uint32_t)pos, 4096, 4,
                        window);
        }
    } else {
        MTL::Buffer* kbf = k_int8() ? dummy_bf16_ : kc_[layer];
        MTL::Buffer* k8 = k_int8() ? kc8_[layer] : dummy_i8_;
        MTL::Buffer* ksc = k_int8() ? ks8_[layer] : dummy_scale_;
        MTL::Buffer* vbf = v_int8() ? dummy_bf16_ : vc_[layer];
        MTL::Buffer* v8 = v_int8() ? vc8_[layer] : dummy_i8_;
        MTL::Buffer* vsc = v_int8() ? vs8_[layer] : dummy_scale_;
        const uint32_t ki = k_int8() ? 1u : 0u;
        if (type == 1 && pos >= 512) {
            const uint32_t splits = 16;
            attn_decode_split_mix(m_, e_, qn_, kbf, k8, ksc, vbf, v8, vsc, pacc_, pm_, pl_,
                                  (uint32_t)pos, kMaxSeq, 4, splits, ki);
            attn_combine(m_, e_, pacc_, pm_, pl_, attn_, splits);
        } else {
            attn_decode_mix(m_, e_, qn_, kbf, k8, ksc, vbf, v8, vsc, attn_,
                            (uint32_t)pos, kMaxSeq, 4, window, ki);
        }
    }
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
          cs_[layer][cur], cs_[layer][1 - cur], h_, 1, 2048);
    cs_cur_[layer] = 1 - cur;
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
        if (T <= 32) gemm_mpp32(m_, e_, A, wbf_, C, T, N, K, off);
        else gemm_mpp(m_, e_, A, wbf_, C, T, N, K, off);
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
    const int type = layer_kind(layer);
    const bool partial = (type == 1);
    const float theta = partial ? 1000000.0f : 10000.0f;
    rope_b(m_, e_, pqn_, pkn_, T, (uint32_t)start, theta, partial);
    const KvMode mode = kv_mode_;
    if (mode == KvMode::Int8) {
        kv_write_b_i8(m_, e_, pkn_, pv_, kc8_[layer], vc8_[layer], ks8_[layer],
                      vs8_[layer], (uint32_t)T, (uint32_t)start, kMaxSeq);
    } else if (mode == KvMode::Bf16) {
        kv_write_b(m_, e_, pkn_, pv_, kc_[layer], vc_[layer], T, (uint32_t)start, 4096);
    } else if (k_int8()) {   // K int8 + V bf16
        kv_write_b_i8(m_, e_, pkn_, pv_, kc8_[layer], dummy_i8_, ks8_[layer],
                      dummy_scale_, (uint32_t)T, (uint32_t)start, kMaxSeq, 1);
        kv_write_b(m_, e_, pkn_, pv_, dummy_bf16_, vc_[layer], T, (uint32_t)start, 4096, 2);
    } else {                 // K bf16 + V int8
        kv_write_b(m_, e_, pkn_, pv_, kc_[layer], dummy_bf16_, T, (uint32_t)start, 4096, 1);
        kv_write_b_i8(m_, e_, pkn_, pv_, dummy_i8_, vc8_[layer], dummy_scale_,
                      vs8_[layer], (uint32_t)T, (uint32_t)start, kMaxSeq, 2);
    }
    const uint32_t seq = (uint32_t)(start + T);
    const uint32_t kbase = (uint32_t)kMaxSeq * kHeadDim * 2;        // bf16 bytes / head
    const uint32_t vbase = (uint32_t)kHeadDim * kMaxSeq * 2;
    const uint32_t kbase8 = (uint32_t)kMaxSeq * kHeadDim;           // int8 bytes / head
    // K scales are [pos][4] and V scales are [4][pos]; both occupy
    // max_seq * 4 * 2 bytes per head.
    const uint32_t sbase8 = (uint32_t)kMaxSeq * 4u * 2u;
    for (uint32_t h = 0; h < 16; ++h) {
        const uint32_t kvh = h / 4u;
        if (k_int8()) {
            gemm_naive_qk_i8(m_, e_, pqn_, kc8_[layer], ks8_[layer], scores_, T, seq, 128,
                             2048, 128, seq, h * 128 * 2, kvh * kbase8, kvh * sbase8, 0);
        } else {
            gemm_naive(m_, e_, pqn_, kc_[layer], scores_, T, seq, 128,
                       2048, 128, seq, h * 128 * 2, kvh * kbase, 0);
        }
        softmax_prefill(m_, e_, scores_, scores_bf_, T, seq, (uint32_t)start,
                        type == 2);
        if (v_int8()) {
            gemm_naive_pv_i8(m_, e_, scores_bf_, vc8_[layer], vs8_[layer], pattn_, T, 128,
                             seq, seq, 4096, 2048, 0, kvh * kbase8, kvh * sbase8,
                             h * 128 * 4);
        } else {
            gemm_naive(m_, e_, scores_bf_, vc_[layer], pattn_, T, 128, seq,
                       seq, 4096, 2048, 0, kvh * vbase, h * 128 * 4);
        }
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
          cs_[layer][cur], cs_[layer][1 - cur], ph_, T, 2048);
    cs_cur_[layer] = 1 - cur;
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

// ---- additive vision entry points -----------------------------------------
// The two functions below are only used by the vision path. They leave the
// text-only prefill/forward/generate code above untouched.

void Engine::embed_tokens(const int* toks, int T, MTL::Buffer* out) {
    const Tensor& eq = w_.get("language_model.model.embed_tokens.weight");
    const Tensor& es = w_.get("language_model.model.embed_tokens.scales");
    const Tensor& eb = w_.get("language_model.model.embed_tokens.biases");
    // All chunks go into one command buffer, so the token ids have to live in a
    // buffer that is not rewritten between dispatches.
    MTL::Buffer* tb = m_.new_buffer((size_t)T * 4);
    uint32_t* tp = (uint32_t*)tb->contents();
    for (int i = 0; i < T; ++i) tp[i] = (uint32_t)toks[i];
    MTL::ComputePipelineState* p = m_.pso("embed8");
    MTL::CommandBuffer* cb = m_.begin();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    const uint32_t K = 2048;
    int done = 0;
    while (done < T) {
        const int n = std::min(128, T - done);
        enc->setComputePipelineState(p);
        enc->setBuffer(w_.buffer(), eq.begin, 0);
        enc->setBuffer(w_.buffer(), es.begin, 1);
        enc->setBuffer(w_.buffer(), eb.begin, 2);
        enc->setBuffer(tb, (NS::UInteger)((size_t)done * 4), 3);
        enc->setBuffer(out, (NS::UInteger)((size_t)done * 2048 * 2), 4);
        enc->setBytes(&K, sizeof(K), 5);
        enc->dispatchThreadgroups(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
        done += n;
    }
    enc->endEncoding();
    m_.submit(cb, true);
    tb->release();
}

void Engine::prefill_embeds(const uint16_t* embeds, int T, bool need_logits) {
    int done = 0;
    while (done < T) {
        const int n = std::min(128, T - done);
        const int start = pos_;
        memcpy(ph_->contents(), embeds + (size_t)done * 2048, (size_t)n * 2048 * 2);
        cb_ = m_.begin();
        e_ = cb_->computeCommandEncoder();
        to_f32(m_, e_, ph_, px_, (uint32_t)(n * 2048));
        for (int l = 0; l < 24; ++l) {
            if (kLayerSuffix[l][0] == 's') p_attn(l, n, start);
            else p_conv(l, n);
        }
        if (need_logits && done + n == T) {
            const Tensor& eq = w_.get("language_model.model.embed_tokens.weight");
            const Tensor& es = w_.get("language_model.model.embed_tokens.scales");
            const Tensor& eb = w_.get("language_model.model.embed_tokens.biases");
            rmsnorm(m_, e_, px_, w_, w_.get("language_model.model.embedding_norm.weight"),
                    ph_, (uint32_t)n, 2048, 1e-5f);
            copy_row(m_, e_, ph_, hlast_, 2048, (uint32_t)(n - 1), 2048);
            gemv8(m_, e_, w_, eq, es, eb, hlast_, logits_, 128000, 2048);
        }
        e_->endEncoding();
        m_.submit(cb_, true);
        pos_ = start + n;
        done += n;
    }
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
            // KV beyond pos_ is never read; skipping the 92 MB clear keeps
            // short-prompt prefill cheap.
        } else {
            for (int k = 0; k < 2; ++k)
                memset(cs_[l][k]->contents(), 0, (size_t)6 * 2048 * 4);
            cs_cur_[l] = 0;
        }
    }
}


void Engine::snapshot(int slot) {
    if (slot < 0 || slot >= kSnapshotSlots) return;
    for (int l = 0; l < 24; ++l) {
        if (kLayerSuffix[l][0] == 's') {
            if (k_int8()) {
                const size_t kb = (size_t)pos_ * kHeadDim;
                const size_t skb = (size_t)pos_ * 4 * 2;   // K scales [pos][4]
                for (int h = 0; h < kKvHeads; ++h) {
                    const size_t off = (size_t)h * kMaxSeq * kHeadDim;
                    if (kb) {
                        memcpy((uint8_t*)snap_kc8_[slot][l]->contents() + off,
                               (const uint8_t*)kc8_[l]->contents() + off, kb);
                    }
                    const size_t ksoff = (size_t)h * kMaxSeq * 4 * 2;
                    if (skb) {
                        memcpy((uint8_t*)snap_ks8_[slot][l]->contents() + ksoff,
                               (const uint8_t*)ks8_[l]->contents() + ksoff, skb);
                    }
                }
            } else {
                for (int h = 0; h < 4; ++h) {
                    const size_t off = (size_t)h * 4096 * 128 * 2;
                    memcpy((uint8_t*)snap_kc_[slot][l]->contents() + off,
                           (const uint8_t*)kc_[l]->contents() + off,
                           (size_t)pos_ * 128 * 2);
                }
            }
            if (v_int8()) {
                const size_t vb = (size_t)pos_;       // int8 bytes per V row
                const size_t sb = (size_t)pos_ * 2;   // scale bytes per group
                for (int h = 0; h < kKvHeads; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const size_t off = ((size_t)h * kHeadDim + d) * kMaxSeq;
                        if (vb) {
                            memcpy((uint8_t*)snap_vc8_[slot][l]->contents() + off,
                                   (const uint8_t*)vc8_[l]->contents() + off, vb);
                        }
                    }
                    for (int g = 0; g < 4; ++g) {
                        const size_t off = ((size_t)h * 4 + g) * kMaxSeq * 2;
                        if (sb) {
                            memcpy((uint8_t*)snap_vs8_[slot][l]->contents() + off,
                                   (const uint8_t*)vs8_[l]->contents() + off, sb);
                        }
                    }
                }
            } else {
                for (int h = 0; h < 4; ++h) {
                    for (int d = 0; d < 128; ++d) {
                        const size_t off = ((size_t)h * 128 + d) * 4096 * 2;
                        memcpy((uint8_t*)snap_vc_[slot][l]->contents() + off,
                               (const uint8_t*)vc_[l]->contents() + off,
                               (size_t)pos_ * 2);
                    }
                }
            }
        } else {
            memcpy(snap_cs_[slot][l]->contents(), cs_[l][cs_cur_[l]]->contents(), (size_t)6 * 2048 * 4);
        }
    }
    snap_pos_[slot] = pos_;
    has_snapshot_[slot] = true;
}

void Engine::restore(int slot) {
    if (slot < 0 || slot >= kSnapshotSlots || !has_snapshot_[slot]) return;
    for (int l = 0; l < 24; ++l) {
        if (kLayerSuffix[l][0] == 's') {
            if (k_int8()) {
                const size_t kb = (size_t)snap_pos_[slot] * kHeadDim;
                const size_t skb = (size_t)snap_pos_[slot] * 4 * 2;
                for (int h = 0; h < kKvHeads; ++h) {
                    const size_t off = (size_t)h * kMaxSeq * kHeadDim;
                    if (kb) {
                        memcpy((uint8_t*)kc8_[l]->contents() + off,
                               (const uint8_t*)snap_kc8_[slot][l]->contents() + off, kb);
                    }
                    const size_t ksoff = (size_t)h * kMaxSeq * 4 * 2;
                    if (skb) {
                        memcpy((uint8_t*)ks8_[l]->contents() + ksoff,
                               (const uint8_t*)snap_ks8_[slot][l]->contents() + ksoff, skb);
                    }
                }
            } else {
                for (int h = 0; h < 4; ++h) {
                    const size_t off = (size_t)h * 4096 * 128 * 2;
                    memcpy((uint8_t*)kc_[l]->contents() + off,
                           (const uint8_t*)snap_kc_[slot][l]->contents() + off,
                           (size_t)snap_pos_[slot] * 128 * 2);
                }
            }
            if (v_int8()) {
                const size_t vb = (size_t)snap_pos_[slot];
                const size_t sb = (size_t)snap_pos_[slot] * 2;
                for (int h = 0; h < kKvHeads; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const size_t off = ((size_t)h * kHeadDim + d) * kMaxSeq;
                        if (vb) {
                            memcpy((uint8_t*)vc8_[l]->contents() + off,
                                   (const uint8_t*)snap_vc8_[slot][l]->contents() + off, vb);
                        }
                    }
                    for (int g = 0; g < 4; ++g) {
                        const size_t off = ((size_t)h * 4 + g) * kMaxSeq * 2;
                        if (sb) {
                            memcpy((uint8_t*)vs8_[l]->contents() + off,
                                   (const uint8_t*)snap_vs8_[slot][l]->contents() + off, sb);
                        }
                    }
                }
            } else {
                for (int h = 0; h < 4; ++h) {
                    for (int d = 0; d < 128; ++d) {
                        const size_t off = ((size_t)h * 128 + d) * 4096 * 2;
                        memcpy((uint8_t*)vc_[l]->contents() + off,
                               (const uint8_t*)snap_vc_[slot][l]->contents() + off,
                               (size_t)snap_pos_[slot] * 2);
                    }
                }
            }
        } else {
            cs_cur_[l] = 0;
            memcpy(cs_[l][0]->contents(), snap_cs_[slot][l]->contents(), (size_t)6 * 2048 * 4);
        }
    }
    pos_ = snap_pos_[slot];
}

void Engine::prefill_ids(const std::vector<int>& ids, bool need_logits) {
    size_t p = 0;
    while (p < ids.size()) {
        int n = (int)std::min<size_t>(128, ids.size() - p);
        prefill_chunk(ids.data() + p, n, pos_, need_logits && (p + n == ids.size()));
        p += (size_t)n;
    }
}

std::vector<int> Engine::decode_loop(const GenConfig& cfg) {
    std::vector<int> out;
    if (pos_ == 0) return out;
    int tok = sample(cfg);
    out.push_back(tok);
    for (int t = 1; t < cfg.max_tokens; ++t) {
        if (tok == 124900) break;
        forward(tok, pos_, true);
        tok = sample(cfg);
        out.push_back(tok);
    }
    return out;
}

std::vector<int> Engine::generate_from_state(const std::vector<int>& suffix, const GenConfig& cfg) {
    prefill_ids(suffix, true);
    return decode_loop(cfg);
}

std::vector<int> Engine::decode_loop_stream(const GenConfig& cfg, const std::function<void(int)>& cb) {
    std::vector<int> out;
    if (pos_ == 0) return out;
    int tok = sample(cfg);
    out.push_back(tok);
    if (cb) cb(tok);
    for (int t = 1; t < cfg.max_tokens; ++t) {
        if (tok == 124900) break;
        forward(tok, pos_, true);
        tok = sample(cfg);
        out.push_back(tok);
        if (cb) cb(tok);
    }
    return out;
}

std::vector<int> Engine::generate_from_state_stream(const std::vector<int>& suffix,
                                                    const GenConfig& cfg,
                                                    const std::function<void(int)>& cb) {
    prefill_ids(suffix, true);
    std::vector<int> out;
    if (pos_ == 0) return out;
    int tok = sample(cfg);
    out.push_back(tok);
    if (cb) cb(tok);
    for (int t = 1; t < cfg.max_tokens; ++t) {
        if (tok == 124900) break;
        forward(tok, pos_, true);
        tok = sample(cfg);
        out.push_back(tok);
        if (cb) cb(tok);
    }
    return out;
}
std::vector<int> Engine::generate(const std::vector<int>& prompt, const GenConfig& cfg) {
    reset();
    auto t0 = std::chrono::steady_clock::now();
    prefill_ids(prompt, true);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<int> out = decode_loop(cfg);
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
