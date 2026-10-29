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

        // ---- packed prefill activations ([kPackRows] rows)
        pxb_ = buf((size_t)kPackRows * kHidden * 4);
        phb_ = buf((size_t)kPackRows * kHidden * 2);
        pqb_ = buf((size_t)kPackRows * kQDim * 4);
        pkb_ = buf((size_t)kPackRows * kKvDim * 4);
        pvb_ = buf((size_t)kPackRows * kKvDim * 4);
        pgb_ = buf((size_t)kPackRows * 16 * 4);
        pqnb_ = buf((size_t)kPackRows * kQDim * 2);
        pknb_ = buf((size_t)kPackRows * kKvDim * 2);
        pab_ = buf((size_t)kPackRows * kHidden * 4);
        pagb_ = buf((size_t)kPackRows * kHidden * 2);
        ppb_ = buf((size_t)kPackRows * kHidden * 4);
        pfgb_ = buf((size_t)kPackRows * kFfn * 4);
        pfub_ = buf((size_t)kPackRows * kFfn * 4);
        pfactb_ = buf((size_t)kPackRows * kFfn * 2);
        pdb_ = buf((size_t)kPackRows * kHidden * 4);
        pbcxb_ = buf((size_t)kPackRows * 3 * kHidden * 4);
        pgatedb_ = buf((size_t)kPackRows * kHidden * 4);
        phlastb_ = buf((size_t)kBmax * kHidden * 2);
        pscores_ = buf((size_t)kPackChunk * kMaxSeq * 4);
        pscores_bf_ = buf((size_t)kPackChunk * kMaxSeq * 2);
        ppzero_ = buf((size_t)kConvState * 4);
        ptok_ = buf((size_t)kPackRows * 4);
        pppos_ = buf((size_t)kPackRows * 4);
        ppslot_ = buf((size_t)kPackRows * 4);
        pprow_ = buf((size_t)kBmax * 4);
        // Optional: keep each request's first-token logits across packed passes so
        // batch-prefill-test can diff them against the sequential path.
        if (getenv("MIORI_BATCH_KEEP_LOGITS")) plogits_ = buf((size_t)kBmax * kVocab * 4);

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
                               logitsb_, embb_, pxb_, phb_, pqb_, pkb_, pvb_, pgb_, pqnb_,
                               pknb_, pab_, pagb_, ppb_, pfgb_, pfub_, pfactb_, pdb_,
                               pbcxb_, pgatedb_, phlastb_, pscores_, pscores_bf_,
                               ppzero_, ptok_, pppos_, ppslot_, pprow_}) {
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

        size_t kv_bytes = 0, cs_bytes = 0, pack_bytes = 0;
        for (int l = 0; l < kLayers; ++l) {
            if (kc_[l]) kv_bytes += 2 * (size_t)kc_[l]->length();
            else cs_bytes += (size_t)cs_[l]->length();
        }
        for (MTL::Buffer* b : {pxb_, phb_, pqb_, pkb_, pvb_, pgb_, pqnb_, pknb_, pab_, pagb_,
                               ppb_, pfgb_, pfub_, pfactb_, pdb_, pbcxb_, pgatedb_,
                               phlastb_, pscores_, pscores_bf_, ppzero_, ptok_, pppos_,
                               ppslot_, pprow_})
            pack_bytes += (size_t)b->length();
        printf("batch: Bmax=%d, slot KV %.1f MB total, conv state %.1f MB, logits %.1f MB\n",
               kBmax, kv_bytes / 1e6, cs_bytes / 1e6,
               (double)kBmax * kVocab * 4 / 1e6);
        printf("batch: packed prefill rows=%d (%.1f MB activations), chunk=%d\n", kPackRows,
               pack_bytes / 1e6, kPackChunk);
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
    if (use_gemv_) {
        gemv4_b(m_, e_, w_, *p.wt, *p.sc, *p.bi, A, C, p.N, p.K, (uint32_t)step_b_);
    } else {
        if (step_b_ <= 32) gemm_mpp32(m_, e_, A, eng_.bf16_buffer(), C, (uint32_t)step_b_, p.N, p.K, p.off);
        else gemm_mpp(m_, e_, A, eng_.bf16_buffer(), C, (uint32_t)step_b_, p.N, p.K, p.off);
    }
}

void BatchEngine::linear_gate(MTL::Buffer* A, MTL::Buffer* C, const Proj& p) {
    if (use_gemv_) {
        gemv4_b(m_, e_, w_, *p.wt, *p.sc, *p.bi, A, C, p.N, p.K, (uint32_t)step_b_);
    } else {
        // N = 16 is not a multiple of 64: the cache holds this tensor row-major.
        gemm_naive(m_, e_, A, eng_.bf16_buffer(), C, (uint32_t)step_b_, p.N, p.K, kHidden, p.K, p.N,
                   0, p.off, 0);
    }
}

void BatchEngine::ffn(int layer) {
    const uint32_t B = (uint32_t)step_b_;
    rmsnorm(m_, e_, xb_, w_, *norm_ffn_[layer], hb_, B, kHidden, 1e-5f);
    linear(hb_, fgb_, lw_[layer].w1);
    linear(hb_, fub_, lw_[layer].w3);
    silu_mul(m_, e_, fgb_, fub_, factb_, B * kFfn);
    linear(factb_, db_, lw_[layer].w2);
    postnorm_add(m_, e_, xb_, db_, w_, *norm_post_ffn_[layer], xb_, B, kHidden, 1e-5f);
}

void BatchEngine::attn_layer(int layer) {
    const uint32_t B = (uint32_t)step_b_;
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
    const uint32_t B = (uint32_t)step_b_;
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

    auto t0 = std::chrono::steady_clock::now();
    // 1. Sequential prefill on the shared Engine (bf16 cache already warm).
    eng_.reset();
    eng_.prefill_ids(prompt, true);
    GenConfig cfg;
    cfg.max_tokens = 1;
    cfg.temperature = 0.0f;
    const int first = eng_.sample(cfg);
    const int pos = eng_.pos();
    auto t1 = std::chrono::steady_clock::now();
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

    auto t2 = std::chrono::steady_clock::now();
    // 3. Copy the live conv state (6 x 2048 floats per conv layer).
    for (int l = 0; l < kLayers; ++l) {
        if (Engine::is_attention_layer(l)) continue;
        float* dst = (float*)cs_[l]->contents() +
                     ((size_t)slot * 2 + (size_t)cs_cur_[l]) * kConvState;
        memcpy(dst, eng_.conv_state(l)->contents(), (size_t)kConvState * 4);
    }

    auto t3 = std::chrono::steady_clock::now();
    if (getenv("MIORI_BATCH_PROFILE"))
        fprintf(stderr, "add_sequence: prefill %.1f ms kv %.1f ms conv %.1f ms (pos %d)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::chrono::duration<double, std::milli>(t3 - t2).count(), pos);
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

// ---- packed multi-request prefill -----------------------------------------
//
// N prompts are concatenated into one [T][2048] activation buffer. Every linear
// layer then runs one MPP GEMM over all T rows; attention, KV writes and the
// conv state update stay per request, on slices of the shared buffers. The
// per-request work reproduces the single-sequence chunking exactly (128-token
// attention chunks, zero conv state at the request start), so the first-token
// logits are bit-identical to Engine::prefill_ids + Engine::sample.

void BatchEngine::linear_p(MTL::Buffer* A, MTL::Buffer* C, const Proj& p, int T) {
    gemm_mpp(m_, e_, A, eng_.bf16_buffer(), C, (uint32_t)T, p.N, p.K, p.off);
}

void BatchEngine::linear_gate_p(MTL::Buffer* A, MTL::Buffer* C, const Proj& p, int T) {
    // N = 16 is not a multiple of 64: the cache holds this tensor row-major.
    gemm_naive(m_, e_, A, eng_.bf16_buffer(), C, (uint32_t)T, p.N, p.K, kHidden,
               p.K, p.N, 0, p.off, 0);
}

void BatchEngine::packed_ffn(int layer, int T) {
    const uint32_t Tu = (uint32_t)T;
    rmsnorm(m_, e_, pxb_, w_, *norm_ffn_[layer], phb_, Tu, kHidden, 1e-5f);
    linear_p(phb_, pfgb_, lw_[layer].w1, T);
    linear_p(phb_, pfub_, lw_[layer].w3, T);
    silu_mul(m_, e_, pfgb_, pfub_, pfactb_, Tu * kFfn);
    linear_p(pfactb_, pdb_, lw_[layer].w2, T);
    postnorm_add(m_, e_, pxb_, pdb_, w_, *norm_post_ffn_[layer], pxb_, Tu, kHidden, 1e-5f);
}

void BatchEngine::packed_attn(int layer, const std::vector<std::vector<int>>& prompts,
                              const std::vector<int>& slots, int T) {
    const uint32_t Tu = (uint32_t)T;
    rmsnorm(m_, e_, pxb_, w_, *norm_op_[layer], phb_, Tu, kHidden, 1e-5f);
    linear_p(phb_, pqb_, lw_[layer].q, T);
    linear_p(phb_, pkb_, lw_[layer].k, T);
    linear_p(phb_, pvb_, lw_[layer].v, T);
    linear_gate_p(phb_, pgb_, lw_[layer].gate, T);
    rmsnorm(m_, e_, pqb_, w_, *norm_q_[layer], pqnb_, Tu * 16, kHeadDim, 1e-5f);
    rmsnorm(m_, e_, pkb_, w_, *norm_k_[layer], pknb_, Tu * 4, kHeadDim, 1e-5f);
    const int kind = Engine::layer_kind(layer);
    const bool partial = (kind == 1);
    const float theta = partial ? 1000000.0f : 10000.0f;
    rope_batch(m_, e_, pqnb_, pknb_, pppos_, Tu, theta, partial);
    kv_write_packed(m_, e_, pknb_, pvb_, kc_[layer], vc_[layer], ppslot_, pppos_, Tu,
                    kMaxSeq, kKvHeads);

    // Per-request attention on its own token range and slot KV. The chunking
    // (and therefore the score-matrix width) matches Engine::prefill_ids.
    int row = 0;
    for (size_t r = 0; r < prompts.size(); ++r) {
        const int len = (int)prompts[r].size();
        const uint32_t slot = (uint32_t)slots[r];
        for (int c = 0; c < len; c += kPackChunk) {
            const int clen = std::min(kPackChunk, len - c);
            const uint32_t seq = (uint32_t)(c + clen);
            const uint32_t row0 = (uint32_t)(row + c);
            for (uint32_t h = 0; h < 16; ++h) {
                const uint32_t kvh = h / 4u;
                const uint32_t kbase =
                    ((slot * kKvHeads + kvh) * kMaxSeq * kHeadDim) * 2u;
                const uint32_t vbase =
                    ((slot * kKvHeads + kvh) * kHeadDim * kMaxSeq) * 2u;
                // scores = q_chunk . K[0..seq)
                gemm_naive(m_, e_, pqnb_, kc_[layer], pscores_, (uint32_t)clen, seq, 128,
                           kHidden, kHeadDim, seq, (row0 * kHidden + h * kHeadDim) * 2u,
                           kbase, 0);
                softmax_prefill(m_, e_, pscores_, pscores_bf_, (uint32_t)clen, seq,
                                (uint32_t)c, kind == 2);
                // out_chunk = P . V[0..seq)
                gemm_naive(m_, e_, pscores_bf_, vc_[layer], pab_, (uint32_t)clen, 128,
                           seq, seq, kMaxSeq, kHidden, 0, vbase,
                           (row0 * kHidden + h * kHeadDim) * 4u);
            }
        }
        row += len;
    }
    gate_mul_b(m_, e_, pab_, pgb_, pagb_, Tu * kHidden);
    linear_p(pagb_, ppb_, lw_[layer].o, T);
    postnorm_add(m_, e_, pxb_, ppb_, w_, *norm_post_op_[layer], pxb_, Tu, kHidden, 1e-5f);
    packed_ffn(layer, T);
}

void BatchEngine::packed_conv(int layer, const std::vector<std::vector<int>>& prompts,
                              const std::vector<int>& slots, int T) {
    const uint32_t Tu = (uint32_t)T;
    rmsnorm(m_, e_, pxb_, w_, *norm_op_[layer], phb_, Tu, kHidden, 1e-5f);
    linear_p(phb_, pbcxb_, lw_[layer].in, T);
    conv_gate(m_, e_, pbcxb_, pgatedb_, Tu, kHidden);
    // Per request: width-7 causal conv with zero state at the request start.
    int row = 0;
    for (size_t r = 0; r < prompts.size(); ++r) {
        const uint32_t len = (uint32_t)prompts[r].size();
        const uint32_t slot = (uint32_t)slots[r];
        conv7_off(m_, e_, pgatedb_, (uint32_t)row * kHidden * 4u, pbcxb_,
                  (uint32_t)row * 3u * kHidden * 4u, w_, *kernel_[layer], ppzero_,
                  cs_[layer], ((slot * 2u + (uint32_t)cs_cur_[layer]) * kConvState) * 4u,
                  phb_, (uint32_t)row * kHidden * 2u, len, kHidden);
        row += (int)len;
    }
    // The packed pass is not a decode step, so the live half does not flip: the
    // state lands in the same half add_sequence() copies into.
    linear_p(phb_, ppb_, lw_[layer].out, T);
    postnorm_add(m_, e_, pxb_, ppb_, w_, *norm_post_op_[layer], pxb_, Tu, kHidden, 1e-5f);
    packed_ffn(layer, T);
}

bool BatchEngine::add_batch_pass(const std::vector<std::vector<int>>& prompts,
                                 const std::vector<int>& slots,
                                 const std::vector<int>& max_new) {
    const int N = (int)prompts.size();
    if (N <= 0 || N > kBmax || slots.size() != prompts.size()) return false;
    int T = 0;
    for (const auto& p : prompts) {
        if (p.empty() || (int)p.size() > kMaxSeq - 1) return false;
        T += (int)p.size();
    }
    if (T <= 0 || T > kPackRows) return false;

    const auto t0 = std::chrono::steady_clock::now();
    uint32_t* toks = (uint32_t*)ptok_->contents();
    uint32_t* poss = (uint32_t*)pppos_->contents();
    uint32_t* slotof = (uint32_t*)ppslot_->contents();
    uint32_t* lastrows = (uint32_t*)pprow_->contents();
    int row = 0;
    for (int r = 0; r < N; ++r) {
        const int len = (int)prompts[r].size();
        for (int j = 0; j < len; ++j) {
            toks[row] = (uint32_t)prompts[r][j];
            poss[row] = (uint32_t)j;
            slotof[row] = (uint32_t)slots[r];
            ++row;
        }
        lastrows[r] = (uint32_t)(row - 1);
    }

    cb_ = m_.begin();
    e_ = cb_->computeCommandEncoder();
    embed8(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, ptok_, phb_, (uint32_t)T, kHidden);
    to_f32(m_, e_, phb_, pxb_, (uint32_t)T * kHidden);
    for (int l = 0; l < kLayers; ++l) {
        if (Engine::is_attention_layer(l)) packed_attn(l, prompts, slots, T);
        else packed_conv(l, prompts, slots, T);
    }
    rmsnorm(m_, e_, pxb_, w_, *norm_emb_, phb_, (uint32_t)T, kHidden, 1e-5f);
    gather_rows(m_, e_, phb_, phlastb_, pprow_, (uint32_t)N, kHidden);
    // Exact 8-bit lm_head for the last row of every request: same kernel and
    // accumulation order as Engine::prefill_chunk's gemv8, 4 rows per dispatch.
    const uint32_t rows4 = (uint32_t)((N + 3) / 4 * 4);
    for (uint32_t r = 0; r < rows4; r += 4) {
        gemv8_b_off(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, phlastb_, logitsb_, kVocab,
                    kHidden, 4, r * kHidden * 2, r * kVocab * 4);
    }
    e_->endEncoding();
    m_.submit(cb_, true);

    const float* lg = (const float*)logitsb_->contents();
    for (int r = 0; r < N; ++r) {
        const float* rowp = lg + (size_t)r * kVocab;
        int best = 0;
        float bv = rowp[0];
        for (int i = 1; i < kVocab; ++i) {
            if (rowp[i] > bv) { bv = rowp[i]; best = i; }
        }
        Slot& sl = slots_[slots[r]];
        sl.active = true;
        sl.finished = false;
        sl.pos = (int)prompts[r].size();
        sl.pending = best;
        sl.max_new = std::max(1, max_new[r]);
        sl.tokens = prompts[r];
        sl.tokens.push_back(best);
        sl.generated.clear();
        sl.generated.push_back(best);
        if (best == kEosId || sl.max_new <= 1) sl.finished = true;
    }
    last_prefill_s_ += std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
    last_prefill_tokens_ += T;
    last_prefill_requests_ += N;
    if (getenv("MIORI_BATCH_PROFILE"))
        fprintf(stderr, "add_batch_pass: %d requests, %d tokens packed in %.2f ms\n", N, T,
                1000.0 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count());
    return true;
}

std::vector<int> BatchEngine::add_batch(const std::vector<std::vector<int>>& prompts,
                                        int max_new_tokens) {
    return add_batch(prompts, std::vector<int>(prompts.size(), std::max(1, max_new_tokens)));
}

std::vector<int> BatchEngine::add_batch(const std::vector<std::vector<int>>& prompts,
                                        const std::vector<int>& max_new) {
    std::vector<int> out(prompts.size(), -1);
    last_packed_.assign(prompts.size(), 0);
    last_packed_row_.assign(prompts.size(), -1);
    if (!ready_) return out;
    last_prefill_s_ = 0.0;
    last_prefill_tokens_ = 0;
    last_prefill_requests_ = 0;
    size_t i = 0;
    while (i < prompts.size()) {
        // Group as many prompts as fit the free slots and the packed row budget.
        std::vector<int> free_slots;
        for (int s = 0; s < kBmax && (int)free_slots.size() < (int)prompts.size(); ++s)
            if (!slots_[s].active) free_slots.push_back(s);
        if (free_slots.empty()) break;
        const std::vector<int>& head = prompts[i];
        if (head.empty() || (int)head.size() > kPackRows || (int)head.size() > kMaxSeq - 1) {
            // Cannot pack this prompt: fall back to the sequential Engine path.
            out[i] = add_sequence(head, max_new[i]);
            ++i;
            continue;
        }
        std::vector<int> idx;
        int total = 0;
        while (i < prompts.size() && idx.size() < free_slots.size()) {
            const std::vector<int>& p = prompts[i];
            if (p.empty() || (int)p.size() > kPackRows || (int)p.size() > kMaxSeq - 1) break;
            if (total + (int)p.size() > kPackRows) break;
            idx.push_back((int)i);
            total += (int)p.size();
            ++i;
        }
        if (idx.empty()) {   // should not happen; be defensive
            out[i] = add_sequence(prompts[i], max_new[i]);
            ++i;
            continue;
        }
        std::vector<int> slots(free_slots.begin(), free_slots.begin() + (long)idx.size());
        std::vector<std::vector<int>> group;
        std::vector<int> group_max(idx.size());
        group.reserve(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            group.push_back(prompts[idx[k]]);
            group_max[k] = max_new[idx[k]];
        }
        if (add_batch_pass(group, slots, group_max)) {
            for (size_t k = 0; k < idx.size(); ++k) {
                out[idx[k]] = slots[k];
                last_packed_[(size_t)idx[k]] = 1;
                last_packed_row_[(size_t)idx[k]] = (int)k;
                if (plogits_)
                    memcpy((float*)plogits_->contents() + (size_t)idx[k] * kVocab,
                           (const float*)logitsb_->contents() + (size_t)k * kVocab,
                           (size_t)kVocab * 4);
            }
        } else {
            for (size_t k = 0; k < idx.size(); ++k)
                out[idx[k]] = add_sequence(group[k], group_max[k]);
        }
    }
    return out;
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
    bool low_slots = true;
    for (int s = 4; s < kBmax; ++s)
        if (slots_[s].active && !slots_[s].finished) low_slots = false;
    use_gemv_ = low_slots && nrun <= 4;
    step_b_ = use_gemv_ ? 4 : kBmax;

    const auto t0 = std::chrono::steady_clock::now();
    cb_ = m_.begin();
    e_ = cb_->computeCommandEncoder();
    embed8(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, tokb_, embb_, step_b_, kHidden);
    to_f32(m_, e_, embb_, xb_, step_b_ * kHidden);
    for (int l = 0; l < kLayers; ++l) {
        if (Engine::is_attention_layer(l)) attn_layer(l);
        else conv_layer(l);
    }
    rmsnorm(m_, e_, xb_, w_, *norm_emb_, hlastb_, step_b_, kHidden, 1e-5f);
    if (use_gemv_) {
        gemv8_b(m_, e_, w_, *emb_q_, *emb_s_, *emb_b_, hlastb_, logitsb_, kVocab,
                kHidden, (uint32_t)step_b_);
    } else {
        gemm_naive(m_, e_, hlastb_, embb16_, logitsb_, step_b_, kVocab, kHidden,
                   kHidden, kHidden, kVocab);
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
