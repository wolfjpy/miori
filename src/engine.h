#pragma once
#include "metal.h"
#include "weights.h"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

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
    std::vector<int> generate_from_state_stream(const std::vector<int>& suffix, const GenConfig& cfg,
                                               const std::function<void(int)>& cb);
    std::vector<int> decode_loop_stream(const GenConfig& cfg, const std::function<void(int)>& cb);
    static constexpr int kSnapshotSlots = 4;
    static constexpr int kKvHeads = 4;
    static constexpr int kHeadDim = 128;
    static constexpr int kMaxSeq = 4096;
    void snapshot(int slot = 0);
    void restore(int slot = 0);
    bool has_snapshot(int slot = 0) const { return has_snapshot_[slot]; }
    MTL::Buffer* alloc_temp(size_t bytes) { return m_.new_buffer(bytes); }
    MTL::Buffer* image_embed_buffer() const { return img_emb_; }
    void reset();
    const float* logits() const;
    int pos() const { return pos_; }

    // ---- optional int8 / mixed-precision KV --------------------------------
    // bf16 remains the default. MIORI_KV_INT8=1 selects full int8;
    // MIORI_KV_MIX=kv selects K int8 + V bf16, MIORI_KV_MIX=vk selects
    // K bf16 + V int8 (the letter is the quantised side). The int8 side uses
    // 32-wide group scales. Settings are read at init; set_kv_mode() switches
    // at runtime for the comparison harness (allocates lazily, does not migrate
    // data, so callers must reset/refill).
    enum class KvMode : int { Bf16 = 0, Int8 = 1, K8V16 = 2, K16V8 = 3 };
    static bool kv_mode_k_int8(KvMode m) { return m == KvMode::Int8 || m == KvMode::K8V16; }
    static bool kv_mode_v_int8(KvMode m) { return m == KvMode::Int8 || m == KvMode::K16V8; }
    static const char* kv_mode_name(KvMode m);
    KvMode kv_mode() const { return kv_mode_; }
    bool kv_int8() const { return kv_mode_ == KvMode::Int8; }
    bool k_int8() const { return kv_mode_k_int8(kv_mode_); }
    bool v_int8() const { return kv_mode_v_int8(kv_mode_); }
    bool set_kv_mode(KvMode m, std::string& err);
    bool set_kv_int8(bool on, std::string& err);
    // Live KV bytes for the active format (11 attention layers; scales included).
    size_t kv_cache_bytes() const;
    MTL::Buffer* kv_k(int layer) const;        // bf16 K when the K side is bf16
    MTL::Buffer* kv_v(int layer) const;        // bf16 V when the V side is bf16
    MTL::Buffer* kv_k8(int layer) const;       // int8 K when the K side is int8
    MTL::Buffer* kv_v8(int layer) const;       // int8 V when the V side is int8
    MTL::Buffer* kv_ks(int layer) const;       // bf16 [4][4096][4] K group scales
    MTL::Buffer* kv_vs(int layer) const;       // bf16 [4][4][4096] V group scales

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
    MTL::Buffer* img_emb_ = nullptr;
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
    MTL::Buffer* snap_kc_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_vc_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_cs_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_kc8_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_vc8_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_ks8_[kSnapshotSlots][24] = {};
    MTL::Buffer* snap_vs8_[kSnapshotSlots][24] = {};
    int snap_pos_[kSnapshotSlots] = {};
    bool has_snapshot_[kSnapshotSlots] = {};
    std::unordered_map<std::string, uint32_t> woff_;
    uint32_t wbf_used_ = 0;
    MTL::Buffer* kc_[24] = {};
    MTL::Buffer* vc_[24] = {};
    MTL::Buffer* kc8_[24] = {};
    MTL::Buffer* vc8_[24] = {};
    MTL::Buffer* ks8_[24] = {};
    MTL::Buffer* vs8_[24] = {};
    KvMode kv_mode_ = KvMode::Bf16;
    // Small scratch handles for the inactive side of a mixed writer/reader
    // (never read or written when masked out).
    MTL::Buffer* dummy_bf16_ = nullptr;
    MTL::Buffer* dummy_i8_ = nullptr;
    MTL::Buffer* dummy_scale_ = nullptr;
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
    bool alloc_k_bf16(std::string& err);
    bool alloc_v_bf16(std::string& err);
    bool alloc_k_int8(std::string& err);
    bool alloc_v_int8(std::string& err);
    bool alloc_kv_for_mode(KvMode m, std::string& err);
};

}  // namespace miori
