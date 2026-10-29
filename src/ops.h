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

void kv_append(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
               MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
               uint32_t pos, uint32_t max_seq, uint32_t kvh);

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
                uint32_t T, uint32_t start, uint32_t max_seq);

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
                 uint32_t kvh);

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

}  // namespace miori
