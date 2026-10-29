#pragma once
#include "metal.h"
#include "weights.h"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace miori {

struct GenConfig {
    int max_tokens = 256;
    float temperature = 0.0f;
    int top_k = 50;
    float top_p = 1.0f;
    uint64_t seed = 1234;
    bool slow_prefill = false;
};

class Engine {
public:
    Engine(Metal& m, Weights& w) : m_(m), w_(w) {}
    bool init(std::string& err);
    // Runs one token through the model. If need_logits, also computes the 128K logits.
    void forward(int token, int pos, bool need_logits);
    int sample(const GenConfig& cfg);
    std::vector<int> generate(const std::vector<int>& prompt, const GenConfig& cfg);
    void prefill_chunk(const int* toks, int T, int start, bool need_logits);
    void prefill_ids(const std::vector<int>& ids, bool need_logits);
    // Additive (vision): embed a token chunk into a caller-supplied shared bf16
    // buffer [T][2048] using the 8-bit embedding table. Chunked internally.
    void embed_tokens(const int* toks, int T, MTL::Buffer* out);
    // Additive (vision): prefill from a caller-supplied [T][2048] bf16
    // embedding matrix (text embeddings with image features spliced in).
    // Starts at pos_ and chunks internally at 128. The text-only path above is
    // unchanged and does not use this.
    void prefill_embeds(const uint16_t* embeds, int T, bool need_logits);
    std::vector<int> decode_loop(const GenConfig& cfg);
    std::vector<int> generate_from_state(const std::vector<int>& suffix, const GenConfig& cfg);
    static constexpr int kSnapshotSlots = 4;
    void snapshot(int slot = 0);
    void restore(int slot = 0);
    bool has_snapshot(int slot = 0) const { return has_snapshot_[slot]; }
    MTL::Buffer* alloc_temp(size_t bytes) { return m_.new_buffer(bytes); }
    void reset();
    const float* logits() const;
    int pos() const { return pos_; }

    // ---- additive accessors for the C2 batch path -------------------------
    // The single-sequence API above is unchanged; these only expose the state
    // a BatchEngine has to read or copy.
    MTL::Buffer* bf16_buffer() const { return wbf_; }
    bool has_weight_offset(const Tensor& t) const { return woff_.count(t.name) != 0; }
    uint32_t weight_offset(const Tensor& t) const;
    uint32_t bf16_bytes() const { return wbf_used_; }
    static bool is_attention_layer(int layer);
    // 1 = full attention, 2 = sliding-window attention, 0 = conv layer.
    static int layer_kind(int layer);
    MTL::Buffer* kv_k(int layer) const;       // [4][4096][128] bf16, null for conv
    MTL::Buffer* kv_v(int layer) const;       // [4][128][4096] bf16, null for conv
    MTL::Buffer* conv_state(int layer) const;  // live [6][2048] f32, null for attn
    double last_prefill_s() const { return last_prefill_s_; }
    double last_decode_s() const { return last_decode_s_; }

private:
    Metal& m_;
    Weights& w_;
    MTL::Buffer* x_ = nullptr;       // [2048] f32 residual
    MTL::Buffer* h_ = nullptr;       // [2048] bf16
    MTL::Buffer* emb_ = nullptr;     // [2048] bf16
    MTL::Buffer* q_ = nullptr;       // [2048] f32
    MTL::Buffer* k_ = nullptr;       // [512] f32
    MTL::Buffer* v_ = nullptr;       // [512] f32
    MTL::Buffer* gate_ = nullptr;    // [16] f32
    MTL::Buffer* qn_ = nullptr;      // [2048] bf16
    MTL::Buffer* kn_ = nullptr;      // [512] bf16
    MTL::Buffer* attn_ = nullptr;    // [2048] f32
    MTL::Buffer* agated_ = nullptr;  // [2048] bf16
    MTL::Buffer* proj_ = nullptr;    // [2048] f32
    MTL::Buffer* norm_ = nullptr;    // [2048] f32
    MTL::Buffer* fg_ = nullptr;      // [7680] f32
    MTL::Buffer* fu_ = nullptr;      // [7680] f32
    MTL::Buffer* fact_ = nullptr;    // [7680] bf16
    MTL::Buffer* down_ = nullptr;    // [2048] f32
    MTL::Buffer* bcx_ = nullptr;     // [6144] f32
    MTL::Buffer* gated_ = nullptr;   // [2048] f32
    MTL::Buffer* conv_y_ = nullptr;  // [2048] f32
    MTL::Buffer* logits_ = nullptr;  // [128000] f32
    MTL::Buffer* tokbuf_ = nullptr;  // [1] u32 (padded)
    MTL::Buffer* px_ = nullptr;
    MTL::Buffer* ph_ = nullptr;
    MTL::Buffer* pq_ = nullptr;
    MTL::Buffer* pk_ = nullptr;
    MTL::Buffer* pv_ = nullptr;
    MTL::Buffer* pgate_ = nullptr;
    MTL::Buffer* pqn_ = nullptr;
    MTL::Buffer* pkn_ = nullptr;
    MTL::Buffer* pattn_ = nullptr;
    MTL::Buffer* pagated_ = nullptr;
    MTL::Buffer* pproj_ = nullptr;
    MTL::Buffer* pnorm_ = nullptr;
    MTL::Buffer* pfg_ = nullptr;
    MTL::Buffer* pfu_ = nullptr;
    MTL::Buffer* pfact_ = nullptr;
    MTL::Buffer* pdown_ = nullptr;
    MTL::Buffer* pbcx_ = nullptr;
    MTL::Buffer* pgated_ = nullptr;
    MTL::Buffer* pconv_y_ = nullptr;
    MTL::Buffer* scores_ = nullptr;
    MTL::Buffer* scores_bf_ = nullptr;
    MTL::Buffer* scratch_ = nullptr;
    MTL::Buffer* hlast_ = nullptr;
    MTL::Buffer* pacc_ = nullptr;
    MTL::Buffer* pm_ = nullptr;
    MTL::Buffer* pl_ = nullptr;
    MTL::Buffer* ptokens_ = nullptr;
    MTL::Buffer* wbf_ = nullptr;
    MTL::Buffer* lm4_ = nullptr;
    MTL::Buffer* lm4s_ = nullptr;
    MTL::Buffer* lm4b_ = nullptr;
    bool lm4_ok_ = false;
    MTL::Buffer* snap_kc_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_vc_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_cs_[kSnapshotSlots][24] = {};
    int snap_pos_[kSnapshotSlots] = {};
    bool has_snapshot_[kSnapshotSlots] = {};
    std::unordered_map<std::string, uint32_t> woff_;
    uint32_t wbf_used_ = 0;
    MTL::Buffer* kc_[24] = {};
    MTL::Buffer* vc_[24] = {};
    MTL::Buffer* cs_[24][2] = {};
    int cs_cur_[24] = {};
    MTL::CommandBuffer* cb_ = nullptr;
    MTL::ComputeCommandEncoder* e_ = nullptr;
    int pos_ = 0;
    double last_prefill_s_ = 0.0;
    double last_decode_s_ = 0.0;

    const Tensor& norm(int layer, const char* which) const;
    void qw(int layer, const char* proj, const Tensor*& q, const Tensor*& s,
            const Tensor*& b, uint32_t& N, uint32_t& K) const;
    void attn_layer(int layer, int pos);
    void conv_layer(int layer, int pos);
    void ffn(int layer);
    void p_gemm(MTL::Buffer* A, uint32_t lda, const Tensor& q, const Tensor& s,
                const Tensor& b, uint32_t N, uint32_t K, MTL::Buffer* C,
                uint32_t ldc, uint32_t T);
    void p_attn(int layer, int T, int start);
    void p_conv(int layer, int T);
    void p_ffn(int layer, int T);
    void p_post(int layer, const char* norm_name, MTL::Buffer* src, int T);
    void post_norm_add(int layer, const char* norm_name, MTL::Buffer* src);
};

}  // namespace miori
