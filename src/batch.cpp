#include "batch.h"
#include "ops.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace miori {

namespace {

constexpr int kEosId = 124900;   // <|im_end|>
constexpr int kPadId = 124893;   // <|pad|>

const char* kNormSuffix[4] = {"operator_norm.weight", "post_operator_norm.weight",
                              "ffn_norm.weight", "post_ffn_norm.weight"};

}  // namespace

bool BatchEngine::init(std::string& err) {
    try {
        if (!eng_.bf16_buffer()) {
            err = "BatchEngine: Engine::init() must complete before BatchEngine::init()";
            return false;
        }
        auto buf = [&](size_t bytes) { return m_.new_buffer(bytes); };

        // ---- activations (64 rows so the MPP tile never reads past the end)
        xb_ = buf((size_t)kRows * kHidden * 4);
        hb_ = buf((size_t)kRows * kHidden * 2);
        qb_ = buf((size_t)kRows * kQDim * 4);
        kb_ = buf((size_t)kRows * kKvDim * 4);
        vb_ = buf((size_t)kRows * kKvDim * 4);
        gb_ = buf((size_t)kRows * 16 * 4);
        qnb_ = buf((size_t)kRows * kQDim * 2);
        knb_ = buf((size_t)kRows * kKvDim * 2);
        ab_ = buf((size_t)kRows * kHidden * 4);
        agb_ = buf((size_t)kRows * kHidden * 2);
        pb_ = buf((size_t)kRows * kHidden * 4);
        fgb_ = buf((size_t)kRows * kFfn * 4);
        fub_ = buf((size_t)kRows * kFfn * 4);
        factb_ = buf((size_t)kRows * kFfn * 2);
        db_ = buf((size_t)kRows * kHidden * 4);
        bcxb_ = buf((size_t)kRows * 3 * kHidden * 4);
        gatedb_ = buf((size_t)kRows * kHidden * 4);
        hlastb_ = buf((size_t)kRows * kHidden * 2);
        logitsb_ = buf((size_t)kBmax * kVocab * 4);
        embb16_ = buf((size_t)kVocab * kHidden * 2);
        embb_ = buf((size_t)kRows * kHidden * 2);
        tokb_ = buf(64 * 4);
        posb_ = buf(64 * 4);

        const size_t slot_kv = (size_t)kKvHeads * kMaxSeq * kHeadDim * 2;  // 4 MB
        for (int l = 0; l < kLayers; ++l) {
            if (Engine::is_attention_layer(l)) {
                kc_[l] = buf(slot_kv * kBmax);
                vc_[l] = buf(slot_kv * kBmax);
                memset(kc_[l]->contents(), 0, slot_kv * kBmax);
                memset(vc_[l]->contents(), 0, slot_kv * kBmax);
            } else {
                cs_[l] = buf((size_t)kBmax * 2 * kConvState * 4);
                memset(cs_[l]->contents(), 0, (size_t)kBmax * 2 * kConvState * 4);
                cs_cur_[l] = 0;
            }
        }

        // Padding rows stay zero for the lifetime of the engine.
        for (MTL::Buffer* b : {xb_, hb_, qb_, kb_, vb_, gb_, qnb_, knb_, ab_, agb_, pb_,
                               fgb_, fub_, factb_, db_, bcxb_, gatedb_, hlastb_,
                               logitsb_, embb_}) {
            memset(b->contents(), 0, (size_t)b->length());
        }
        memset(posb_->contents(), 0, 64 * 4);

        err.clear();
        if (!load_weights(err)) return false;
        {
            cb_ = m_.begin();
            e_ = cb_->computeCommandEncoder();
            dequant8(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, embb16_, kVocab, kHidden);
            e_->endEncoding();
            m_.submit(cb_, true);
        }

        size_t kv_bytes = 0, cs_bytes = 0;
        for (int l = 0; l < kLayers; ++l) {
            if (kc_[l]) kv_bytes += 2 * (size_t)kc_[l]->length();
            else cs_bytes += (size_t)cs_[l]->length();
        }
        printf("batch: Bmax=%d, slot KV %.1f MB total, conv state %.1f MB, logits %.1f MB\n",
               kBmax, kv_bytes / 1e6, cs_bytes / 1e6,
               (double)kBmax * kVocab * 4 / 1e6);
        ready_ = true;
    } catch (const std::exception& e) {
        err = std::string("BatchEngine: ") + e.what();
        return false;
    }
    return true;
}

bool BatchEngine::load_weights(std::string& err) {
    for (int l = 0; l < kLayers; ++l) {
        const std::string base = "language_model.model.layers." + std::to_string(l) + ".";
        norm_op_[l] = &w_.get(base + kNormSuffix[0]);
        norm_post_op_[l] = &w_.get(base + kNormSuffix[1]);
        norm_ffn_[l] = &w_.get(base + kNormSuffix[2]);
        norm_post_ffn_[l] = &w_.get(base + kNormSuffix[3]);
        auto cache = [&](const char* proj, Proj& p, bool mpp) -> bool {
            const Tensor& q = w_.get(base + proj + ".weight");
            if (!eng_.has_weight_offset(q)) {
                err = "BatchEngine: weight not in bf16 cache: " + q.name;
                return false;
            }
            p.off = eng_.weight_offset(q);
            p.N = (uint32_t)q.rows();
            p.K = (uint32_t)(q.cols() * 8);
            p.wt = &q;
            p.sc = &w_.get(base + proj + ".scales");
            p.bi = &w_.get(base + proj + ".biases");
            if (mpp && (p.N % 64u) != 0u) {
                err = "BatchEngine: " + q.name + " N is not a multiple of 64";
                return false;
            }
            return true;
        };
        if (Engine::is_attention_layer(l)) {
            if (!cache("self_attn.q_proj", lw_[l].q, true)) return false;
            if (!cache("self_attn.k_proj", lw_[l].k, true)) return false;
            if (!cache("self_attn.v_proj", lw_[l].v, true)) return false;
            if (!cache("self_attn.gate_proj", lw_[l].gate, false)) return false;
            if (!cache("self_attn.o_proj", lw_[l].o, true)) return false;
            norm_q_[l] = &w_.get(base + "self_attn.q_norm.weight");
            norm_k_[l] = &w_.get(base + "self_attn.k_norm.weight");
        } else {
            if (!cache("conv.in_proj", lw_[l].in, true)) return false;
            if (!cache("conv.out_proj", lw_[l].out, true)) return false;
            kernel_[l] = &w_.get(base + "conv.conv.weight");
        }
        if (!cache("feed_forward.w1", lw_[l].w1, true)) return false;
        if (!cache("feed_forward.w3", lw_[l].w3, true)) return false;
        if (!cache("feed_forward.w2", lw_[l].w2, true)) return false;
    }
    emb_q_ = &w_.get("language_model.model.embed_tokens.weight");
    emb_s_ = &w_.get("language_model.model.embed_tokens.scales");
    emb_b_ = &w_.get("language_model.model.embed_tokens.biases");
    norm_emb_ = &w_.get("language_model.model.embedding_norm.weight");
    return true;
}

void BatchEngine::linear(MTL::Buffer* A, MTL::Buffer* C, const Proj& p) {
    if (kernels_ == Kernels::Gemv4) {
        gemv4_b(m_, e_, w_, *p.wt, *p.sc, *p.bi, A, C, p.N, p.K, (uint32_t)kBmax);
    } else {
        gemm_mpp(m_, e_, A, eng_.bf16_buffer(), C, kRows, p.N, p.K, p.off);
    }
}

void BatchEngine::linear_gate(MTL::Buffer* A, MTL::Buffer* C, const Proj& p) {
    if (kernels_ == Kernels::Gemv4) {
        gemv4_b(m_, e_, w_, *p.wt, *p.sc, *p.bi, A, C, p.N, p.K, (uint32_t)kBmax);
    } else {
        // N = 16 is not a multiple of 64: the cache holds this tensor row-major.
        gemm_naive(m_, e_, A, eng_.bf16_buffer(), C, kRows, p.N, p.K, kHidden, p.K, p.N,
                   0, p.off, 0);
    }
}

void BatchEngine::ffn(int layer) {
    const uint32_t B = (uint32_t)kBmax;
    rmsnorm(m_, e_, xb_, w_, *norm_ffn_[layer], hb_, B, kHidden, 1e-5f);
    linear(hb_, fgb_, lw_[layer].w1);
    linear(hb_, fub_, lw_[layer].w3);
    silu_mul(m_, e_, fgb_, fub_, factb_, B * kFfn);
    linear(factb_, db_, lw_[layer].w2);
    postnorm_add(m_, e_, xb_, db_, w_, *norm_post_ffn_[layer], xb_, B, kHidden, 1e-5f);
}

void BatchEngine::attn_layer(int layer) {
    const uint32_t B = (uint32_t)kBmax;
    rmsnorm(m_, e_, xb_, w_, *norm_op_[layer], hb_, B, kHidden, 1e-5f);
    linear(hb_, qb_, lw_[layer].q);
    linear(hb_, kb_, lw_[layer].k);
    linear(hb_, vb_, lw_[layer].v);
    linear_gate(hb_, gb_, lw_[layer].gate);
    rmsnorm(m_, e_, qb_, w_, *norm_q_[layer], qnb_, B * 16, kHeadDim, 1e-5f);
    rmsnorm(m_, e_, kb_, w_, *norm_k_[layer], knb_, B * kKvHeads, kHeadDim, 1e-5f);
    const int kind = Engine::layer_kind(layer);
    const bool partial = (kind == 1);
    const float theta = partial ? 1000000.0f : 10000.0f;
    rope_batch(m_, e_, qnb_, knb_, posb_, B, theta, partial);
    kv_append_b(m_, e_, knb_, vb_, kc_[layer], vc_[layer], posb_, B, kMaxSeq, kKvHeads);
    attn_decode_b(m_, e_, qnb_, kc_[layer], vc_[layer], ab_, posb_, B, kMaxSeq, kKvHeads,
                  kind == 2 ? 256u : 0u);
    gate_mul_b(m_, e_, ab_, gb_, agb_, B * kHidden);
    linear(agb_, pb_, lw_[layer].o);
    postnorm_add(m_, e_, xb_, pb_, w_, *norm_post_op_[layer], xb_, B, kHidden, 1e-5f);
    ffn(layer);
}

void BatchEngine::conv_layer(int layer) {
    const uint32_t B = (uint32_t)kBmax;
    rmsnorm(m_, e_, xb_, w_, *norm_op_[layer], hb_, B, kHidden, 1e-5f);
    linear(hb_, bcxb_, lw_[layer].in);
    conv_gate(m_, e_, bcxb_, gatedb_, B, kHidden);
    const uint32_t cur = (uint32_t)cs_cur_[layer];
    conv7_b(m_, e_, gatedb_, bcxb_, w_, *kernel_[layer], cs_[layer], hb_, cur, B, kHidden);
    cs_cur_[layer] = 1 - cs_cur_[layer];
    linear(hb_, pb_, lw_[layer].out);
    postnorm_add(m_, e_, xb_, pb_, w_, *norm_post_op_[layer], xb_, B, kHidden, 1e-5f);
    ffn(layer);
}

int BatchEngine::add_sequence(const std::vector<int>& prompt, int max_new_tokens) {
    if (!ready_ || prompt.empty()) return -1;
    int slot = -1;
    for (int s = 0; s < kBmax; ++s) {
        if (!slots_[s].active) { slot = s; break; }
    }
    if (slot < 0) return -1;

    // 1. Sequential prefill on the shared Engine (bf16 cache already warm).
    eng_.reset();
    eng_.prefill_ids(prompt, true);
    GenConfig cfg;
    cfg.max_tokens = 1;
    cfg.temperature = 0.0f;
    const int first = eng_.sample(cfg);
    const int pos = eng_.pos();
    if (pos >= kMaxSeq - 1) {
        fprintf(stderr, "batch: prompt too long (%d >= %d)\n", pos, kMaxSeq - 1);
        return -1;
    }

    // 2. Copy the used KV positions per head into the slot's cache.
    const size_t kb = (size_t)pos * kHeadDim * 2;   // bytes used per head
    const size_t vs = (size_t)pos * 2;              // bytes used per V row
    for (int l = 0; l < kLayers; ++l) {
        if (!Engine::is_attention_layer(l)) continue;
        const uint8_t* sk = (const uint8_t*)eng_.kv_k(l)->contents();
        const uint8_t* sv = (const uint8_t*)eng_.kv_v(l)->contents();
        uint8_t* dk = (uint8_t*)kc_[l]->contents();
        uint8_t* dv = (uint8_t*)vc_[l]->contents();
        const size_t slot_off = (size_t)slot * kKvHeads * kMaxSeq * kHeadDim * 2;
        for (int h = 0; h < kKvHeads; ++h) {
            const size_t head_off = (size_t)h * kMaxSeq * kHeadDim * 2;
            if (kb) memcpy(dk + slot_off + head_off, sk + head_off, kb);
            for (int d = 0; d < kHeadDim; ++d) {
                const size_t row_off = ((size_t)h * kHeadDim + d) * kMaxSeq * 2;
                if (vs) memcpy(dv + slot_off + row_off, sv + row_off, vs);
            }
        }
    }

    // 3. Copy the live conv state (6 x 2048 floats per conv layer).
    for (int l = 0; l < kLayers; ++l) {
        if (Engine::is_attention_layer(l)) continue;
        float* dst = (float*)cs_[l]->contents() +
                     ((size_t)slot * 2 + (size_t)cs_cur_[l]) * kConvState;
        memcpy(dst, eng_.conv_state(l)->contents(), (size_t)kConvState * 4);
    }

    Slot& sl = slots_[slot];
    sl.active = true;
    sl.finished = false;
    sl.pos = pos;
    sl.pending = first;
    sl.max_new = std::max(1, max_new_tokens);
    sl.tokens = prompt;
    sl.tokens.push_back(first);
    sl.generated.clear();
    sl.generated.push_back(first);
    if (first == kEosId || sl.max_new <= 1) sl.finished = true;
    return slot;
}

int BatchEngine::decode_step() {
    uint32_t* toks = (uint32_t*)tokb_->contents();
    uint32_t* poss = (uint32_t*)posb_->contents();
    int nrun = 0;
    for (int s = 0; s < kBmax; ++s) {
        Slot& sl = slots_[s];
        if (sl.active && !sl.finished && sl.pos >= kMaxSeq - 1) sl.finished = true;
        if (sl.active && !sl.finished) {
            toks[s] = (uint32_t)sl.pending;
            poss[s] = (uint32_t)sl.pos;
            ++nrun;
        } else {
            toks[s] = (uint32_t)kPadId;
            poss[s] = 0;
        }
    }
    if (!nrun) return 0;

    const auto t0 = std::chrono::steady_clock::now();
    cb_ = m_.begin();
    e_ = cb_->computeCommandEncoder();
    embed8(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, tokb_, embb_, kBmax, kHidden);
    to_f32(m_, e_, embb_, xb_, kBmax * kHidden);
    for (int l = 0; l < kLayers; ++l) {
        if (Engine::is_attention_layer(l)) attn_layer(l);
        else conv_layer(l);
    }
    rmsnorm(m_, e_, xb_, w_, *norm_emb_, hlastb_, kBmax, kHidden, 1e-5f);
    if (kernels_ == Kernels::Gemv4) {
        // One pass over the 8-bit embedding matrix for every slot.
        gemv8_b(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, hlastb_, logitsb_, kVocab,
                kHidden, (uint32_t)kBmax);
    } else if (kBmax > 4) {
        gemm_naive(m_, e_, hlastb_, embb16_, logitsb_, kBmax, kVocab, kHidden,
                   kHidden, kHidden, kVocab);
    } else {
        for (int s = 0; s < kBmax; ++s) {
            if (!slots_[s].active || slots_[s].finished) continue;
            gemv8_off(m_, e_, w_.buffer(), (uint32_t)emb_q_->begin, (uint32_t)emb_s_->begin,
                      (uint32_t)emb_b_->begin, hlastb_, logitsb_, kVocab, kHidden,
                      (uint32_t)s * kHidden * 2, (uint32_t)s * kVocab * 4);
        }
    }
    e_->endEncoding();
    m_.submit(cb_, true);

    // Greedy sampling per slot on the CPU.
    const float* lg = (const float*)logitsb_->contents();
    int produced = 0;
    for (int s = 0; s < kBmax; ++s) {
        Slot& sl = slots_[s];
        if (!sl.active || sl.finished) continue;
        const float* row = lg + (size_t)s * kVocab;
        int best = 0;
        float bv = row[0];
        for (int i = 1; i < kVocab; ++i) {
            if (row[i] > bv) { bv = row[i]; best = i; }
        }
        sl.generated.push_back(best);
        sl.tokens.push_back(best);
        sl.pos += 1;
        sl.pending = best;
        ++produced;
        ++total_generated_;
        if (best == kEosId || (int)sl.generated.size() >= sl.max_new) sl.finished = true;
    }
    const auto t1 = std::chrono::steady_clock::now();
    total_decode_s_ += std::chrono::duration<double>(t1 - t0).count();
    ++steps_;
    return produced;
}

int BatchEngine::active() const {
    int n = 0;
    for (int s = 0; s < kBmax; ++s) if (slots_[s].active) ++n;
    return n;
}

int BatchEngine::running() const {
    int n = 0;
    for (int s = 0; s < kBmax; ++s) if (slots_[s].active && !slots_[s].finished) ++n;
    return n;
}

}  // namespace miori
