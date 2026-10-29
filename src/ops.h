#pragma once
#include "metal.h"
#include "weights.h"
#include <cstdint>

namespace miori {

void gemv4(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
           const Tensor& q, const Tensor& s, const Tensor& b,
           MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K);

void gemv8(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
           const Tensor& q, const Tensor& s, const Tensor& b,
           MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K);

void rmsnorm(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
             const Weights& w, const Tensor& gamma, MTL::Buffer* y,
             uint32_t rows, uint32_t n, float eps);

void add(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* a, MTL::Buffer* b,
         MTL::Buffer* out, uint32_t n);

void silu_mul(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* gate,
              MTL::Buffer* up, MTL::Buffer* out, uint32_t n);

void dequant4(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
              const Tensor& q, const Tensor& s, const Tensor& b,
              MTL::Buffer* out, uint32_t N, uint32_t K, uint32_t boff = 0);

void gemm_naive(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
                MTL::Buffer* B, MTL::Buffer* C, uint32_t M, uint32_t N, uint32_t K,
                uint32_t lda, uint32_t ldb, uint32_t ldc,
                uint32_t aoff = 0, uint32_t boff = 0, uint32_t coff = 0);

void embed8(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
            const Tensor& q, const Tensor& s, const Tensor& b,
            MTL::Buffer* tokens, MTL::Buffer* out, uint32_t T, uint32_t K);

void conv_gate(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* bcx,
               MTL::Buffer* gated, uint32_t T, uint32_t H);

void conv7(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* gated,
           MTL::Buffer* bcx, const Weights& w, const Tensor& kernel,
           MTL::Buffer* state, MTL::Buffer* state_out, MTL::Buffer* y,
           uint32_t T, uint32_t H);

// mode: 1 = write K only, 2 = write V only, 3 = both (default). Used by the
// mixed K/V precision modes to reuse the fused writers with one side masked.
void kv_append(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
               MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
               uint32_t pos, uint32_t max_seq, uint32_t kvh, uint32_t mode = 3);

void rmsnorm_f32(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
                 const Weights& w, const Tensor& gamma, MTL::Buffer* y,
                 uint32_t rows, uint32_t n, float eps);

void to_bf16(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
             MTL::Buffer* y, uint32_t n);

void gate_mul(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* attn,
              MTL::Buffer* gate, MTL::Buffer* out, uint32_t n);

void rope(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
          MTL::Buffer* k, uint32_t pos, float theta, bool partial);

void attn_decode(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                 MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* out,
                 uint32_t pos, uint32_t max_seq, uint32_t kvh, uint32_t window);

void to_f32(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
            MTL::Buffer* y, uint32_t n);

void rope_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
            MTL::Buffer* k, uint32_t T, uint32_t start_pos, float theta, bool partial);

void gate_mul_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* attn,
                MTL::Buffer* gate, MTL::Buffer* out, uint32_t total);

void kv_write_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                uint32_t T, uint32_t start, uint32_t max_seq, uint32_t mode = 3);

void softmax_prefill(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* scores,
                     MTL::Buffer* out, uint32_t T, uint32_t seq, uint32_t start_pos,
                     bool sliding);

void copy_row(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* src,
              MTL::Buffer* dst, uint32_t stride, uint32_t row, uint32_t n);

void dequant_tile(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
                  const Tensor& q, const Tensor& s, const Tensor& b,
                  MTL::Buffer* out, uint32_t N, uint32_t K, uint32_t boff = 0);

void gemm_mpp(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
              MTL::Buffer* B, MTL::Buffer* C, uint32_t M, uint32_t N, uint32_t K,
              uint32_t boff = 0);

// Same tensor-core GEMM with byte offsets into A, B and C. A must be
// contiguous [M][K] from aoff, C contiguous [M][N] from coff, and B must be in
// the [N/64][K][64] tile-major layout from boff. N must be a multiple of 64.
void gemm_mpp_off(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
                  MTL::Buffer* B, MTL::Buffer* C, uint32_t M, uint32_t N, uint32_t K,
                  uint32_t aoff, uint32_t boff, uint32_t coff);

void gemv4x4(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
             const Tensor& q, const Tensor& s, const Tensor& b,
             MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K);

void postnorm_add(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
                  MTL::Buffer* src, const Weights& w, const Tensor& gamma,
                  MTL::Buffer* out, uint32_t rows, uint32_t n, float eps);

void gemv4_raw(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
               MTL::Buffer* s, MTL::Buffer* b, MTL::Buffer* x, MTL::Buffer* y,
               uint32_t N, uint32_t K);

void attn_decode_split(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                       MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* pacc,
                       MTL::Buffer* pm, MTL::Buffer* pl, uint32_t pos,
                       uint32_t max_seq, uint32_t kvh, uint32_t splits);

void attn_combine(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* pacc,
                  MTL::Buffer* pm, MTL::Buffer* pl, MTL::Buffer* out,
                  uint32_t splits);

void gemv4x2(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
             const Tensor& q, const Tensor& s, const Tensor& b,
             MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K);

// ---- C2 batch decode -------------------------------------------------------

// gemv8 with explicit byte offsets into x (bf16) and y (f32), used for the
// per-slot lm_head on a shared [B][2048] / [B][128000] buffer. The 8-bit weight
// tensors all live in one MTLBuffer, so q/s/b are given as byte offsets.
void gemv8_off(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* wbuf,
               uint32_t qoff, uint32_t soff, uint32_t boff, MTL::Buffer* x,
               MTL::Buffer* y, uint32_t N, uint32_t K, uint32_t xoff,
               uint32_t yoff);

// RoPE for B rows with a per-slot position buffer.
void rope_batch(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                MTL::Buffer* k, MTL::Buffer* positions, uint32_t B,
                float theta, bool partial);

// Append one token per slot at positions[slot].
void kv_append_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                 MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                 MTL::Buffer* positions, uint32_t B, uint32_t max_seq,
                 uint32_t kvh, uint32_t mode = 3);

// Decode attention for B slots, grid B*kvh, per-slot positions and window.
void attn_decode_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                   MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* out,
                   MTL::Buffer* positions, uint32_t B, uint32_t max_seq,
                   uint32_t kvh, uint32_t window);

// Causal 7-tap conv for B slots, one token each, double-buffered state.
void conv7_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* gated,
             MTL::Buffer* bcx, const Weights& w, const Tensor& kernel,
             MTL::Buffer* state, MTL::Buffer* y, uint32_t cur, uint32_t B,
             uint32_t H);

// 4-bit GEMV over B <= 4 input rows with one pass over the weight row.
void gemv4_b(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
             const Tensor& q, const Tensor& s, const Tensor& b, MTL::Buffer* x,
             MTL::Buffer* y, uint32_t N, uint32_t K, uint32_t B);

// 8-bit GEMV over B <= 4 input rows (batched lm_head).
void gemv8_b(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
             const Tensor& q, const Tensor& s, const Tensor& b, MTL::Buffer* x,
             MTL::Buffer* y, uint32_t N, uint32_t K, uint32_t B);

// Same kernel with byte offsets into x ([B][K] bf16) and y ([B][N] f32). The
// kernel always computes 4 rows, so B must be 4 and both buffers must hold
// 4 rows from the offset.
void gemv8_b_off(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
                 const Tensor& q, const Tensor& s, const Tensor& b, MTL::Buffer* x,
                 MTL::Buffer* y, uint32_t N, uint32_t K, uint32_t B, uint32_t xoff,
                 uint32_t yoff);

// ---- packed multi-request prefill ------------------------------------------

// Write a packed [T][512] K (bf16, post-RoPE) / V (f32) chunk into per-request
// KV slots. slots[r] / positions[r] give the destination slot and position of
// packed row r. Byte-identical to the single-sequence kv_write_b layout.
void kv_write_packed(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                     MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                     MTL::Buffer* slots, MTL::Buffer* positions, uint32_t T,
                     uint32_t max_seq, uint32_t kvh, uint32_t mode = 3);

// conv7 for one request of a packed batch, with byte offsets into gated [T][H],
// bcx [T][3H], y [T][H] and state_out [6][H]. `state` is the 6-token window
// entering the request (zero for a fresh prompt).
void conv7_off(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* gated,
               uint32_t goff, MTL::Buffer* bcx, uint32_t boff, const Weights& w,
               const Tensor& kernel, MTL::Buffer* state, MTL::Buffer* state_out,
               uint32_t sooff, MTL::Buffer* y, uint32_t yoff, uint32_t T, uint32_t H);

// dst[r][i] = src[rows[r]][i] over bf16 [N][K] rows (last-token gather).
void gather_rows(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* src,
                 MTL::Buffer* dst, MTL::Buffer* rows, uint32_t N, uint32_t K);

void dequant8(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
              const Tensor& q, const Tensor& s, const Tensor& b,
              MTL::Buffer* out, uint32_t N, uint32_t K);

void gemm_mpp32(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
                MTL::Buffer* B, MTL::Buffer* C, uint32_t M, uint32_t N, uint32_t K,
                uint32_t boff = 0);

// ---- optional symmetric int8 KV cache (MIORI_KV_INT8=1) --------------------
//
// K: int8 [h][max_seq][128], ks: bf16 [h][max_seq][4]
// V: int8 [h][128][max_seq], vs: bf16 [h][4][max_seq]
// One bf16 scale per (kv_head, token, 32-wide d group), max_abs(group)/127, no
// zero point. The bf16 kernels above are untouched; callers pick one set or
// the other. Four groups cover the 128-element head, so V's [4][max_seq]
// per-head block and K's [max_seq][4] block have the same byte size.

void kv_append_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                  MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                  MTL::Buffer* ks, MTL::Buffer* vs, uint32_t pos, uint32_t max_seq,
                  uint32_t kvh, uint32_t mode = 3);

void kv_write_b_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                   MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                   MTL::Buffer* ks, MTL::Buffer* vs, uint32_t T, uint32_t start,
                   uint32_t max_seq, uint32_t mode = 3);

void attn_decode_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                    MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* ks,
                    MTL::Buffer* vs, MTL::Buffer* out, uint32_t pos, uint32_t max_seq,
                    uint32_t kvh, uint32_t window);

void attn_decode_split_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                          MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* ks,
                          MTL::Buffer* vs, MTL::Buffer* pacc, MTL::Buffer* pm,
                          MTL::Buffer* pl, uint32_t pos, uint32_t max_seq, uint32_t kvh,
                          uint32_t splits);

// Prefill attention with K/V dequantised on the fly:
//   QK: C[M][N] = A[M][K] * B[N][K]^T, B int8 scaled per (row N, 32-wide K group).
//   PV: C[M][N] = A[M][K] * B[N][K]^T, B int8 scaled per (32-wide N group, column K).
// Byte offsets select the head / slot slice, exactly like gemm_naive.
void gemm_naive_qk_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
                      MTL::Buffer* B, MTL::Buffer* bs, MTL::Buffer* C, uint32_t M,
                      uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc,
                      uint32_t aoff = 0, uint32_t boff = 0, uint32_t bsoff = 0,
                      uint32_t coff = 0);

void gemm_naive_pv_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
                      MTL::Buffer* B, MTL::Buffer* bs, MTL::Buffer* C, uint32_t M,
                      uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc,
                      uint32_t aoff = 0, uint32_t boff = 0, uint32_t bsoff = 0,
                      uint32_t coff = 0);

// Batch (leading slot dimension) int8 writers / decode attention.
void kv_append_b_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                    MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                    MTL::Buffer* ks, MTL::Buffer* vs, MTL::Buffer* positions,
                    uint32_t B, uint32_t max_seq, uint32_t kvh, uint32_t mode = 3);

void attn_decode_b_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                      MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* ks,
                      MTL::Buffer* vs, MTL::Buffer* out, MTL::Buffer* positions,
                      uint32_t B, uint32_t max_seq, uint32_t kvh, uint32_t window);

void kv_write_packed_i8(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                        MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                        MTL::Buffer* ks, MTL::Buffer* vs, MTL::Buffer* slots,
                        MTL::Buffer* positions, uint32_t T, uint32_t max_seq,
                        uint32_t kvh, uint32_t mode = 3);

// ---- mixed K/V precision ---------------------------------------------------
// k_int8 selects K int8 + V bf16 (true) or K bf16 + V int8 (false). The int8
// side uses the 32-wide group scales; the bf16 side is the normal cache. The
// pointer for the inactive side is ignored (pass any valid buffer).
void attn_decode_mix(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                     MTL::Buffer* kc_bf, MTL::Buffer* kc8, MTL::Buffer* ks,
                     MTL::Buffer* vc_bf, MTL::Buffer* vc8, MTL::Buffer* vs,
                     MTL::Buffer* out, uint32_t pos, uint32_t max_seq, uint32_t kvh,
                     uint32_t window, uint32_t k_int8);

void attn_decode_split_mix(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                           MTL::Buffer* kc_bf, MTL::Buffer* kc8, MTL::Buffer* ks,
                           MTL::Buffer* vc_bf, MTL::Buffer* vc8, MTL::Buffer* vs,
                           MTL::Buffer* pacc, MTL::Buffer* pm, MTL::Buffer* pl,
                           uint32_t pos, uint32_t max_seq, uint32_t kvh,
                           uint32_t splits, uint32_t k_int8);

void attn_decode_b_mix(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                       MTL::Buffer* kc_bf, MTL::Buffer* kc8, MTL::Buffer* ks,
                       MTL::Buffer* vc_bf, MTL::Buffer* vc8, MTL::Buffer* vs,
                       MTL::Buffer* out, MTL::Buffer* positions, uint32_t B,
                       uint32_t max_seq, uint32_t kvh, uint32_t window,
                       uint32_t k_int8);


}  // namespace miori
