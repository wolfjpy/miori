#include "ops.h"
#include <algorithm>

namespace miori {


void gemv4(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
           const Tensor& q, const Tensor& s, const Tensor& b,
           MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K) {
    auto p = m.pso("gemv4");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(x, 0, 3);
    e->setBuffer(y, 0, 4);
    e->setBytes(&N, sizeof(N), 5);
    e->setBytes(&K, sizeof(K), 6);
    e->dispatchThreadgroups(MTL::Size::Make((N + 15) / 16, 1, 1),
                            MTL::Size::Make(512, 1, 1));
}

void gemv8(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
           const Tensor& q, const Tensor& s, const Tensor& b,
           MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K) {
    auto p = m.pso("gemv8");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(x, 0, 3);
    e->setBuffer(y, 0, 4);
    e->setBytes(&N, sizeof(N), 5);
    e->setBytes(&K, sizeof(K), 6);
    e->dispatchThreadgroups(MTL::Size::Make((N + 15) / 16, 1, 1),
                            MTL::Size::Make(512, 1, 1));
}

void rmsnorm(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
             const Weights& w, const Tensor& gamma, MTL::Buffer* y,
             uint32_t rows, uint32_t n, float eps) {
    auto p = m.pso("rmsnorm");
    e->setComputePipelineState(p);
    e->setBuffer(x, 0, 0);
    e->setBuffer(w.buffer(), gamma.begin, 1);
    e->setBuffer(y, 0, 2);
    e->setBytes(&n, sizeof(n), 3);
    e->setBytes(&eps, sizeof(eps), 4);
    e->dispatchThreadgroups(MTL::Size::Make(rows, 1, 1), MTL::Size::Make(256, 1, 1));
}

void add(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* a, MTL::Buffer* b,
         MTL::Buffer* out, uint32_t n) {
    auto p = m.pso("add");
    e->setComputePipelineState(p);
    e->setBuffer(a, 0, 0);
    e->setBuffer(b, 0, 1);
    e->setBuffer(out, 0, 2);
    e->setBytes(&n, sizeof(n), 3);
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
}

void silu_mul(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* gate,
              MTL::Buffer* up, MTL::Buffer* out, uint32_t n) {
    auto p = m.pso("silu_mul");
    e->setComputePipelineState(p);
    e->setBuffer(gate, 0, 0);
    e->setBuffer(up, 0, 1);
    e->setBuffer(out, 0, 2);
    e->setBytes(&n, sizeof(n), 3);
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
}

void dequant4(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
              const Tensor& q, const Tensor& s, const Tensor& b,
              MTL::Buffer* out, uint32_t N, uint32_t K, uint32_t boff) {
    auto p = m.pso("dequant4");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(out, boff, 3);
    e->setBytes(&N, sizeof(N), 4);
    e->setBytes(&K, sizeof(K), 5);
    uint64_t total = (uint64_t)N * (K / 32);
    e->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
}

void gemm_naive(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
                MTL::Buffer* B, MTL::Buffer* C, uint32_t M, uint32_t N, uint32_t K,
                uint32_t lda, uint32_t ldb, uint32_t ldc,
                uint32_t aoff, uint32_t boff, uint32_t coff) {
    auto p = m.pso("gemm_naive");
    e->setComputePipelineState(p);
    e->setBuffer(A, aoff, 0);
    e->setBuffer(B, boff, 1);
    e->setBuffer(C, coff, 2);
    e->setBytes(&M, sizeof(M), 3);
    e->setBytes(&N, sizeof(N), 4);
    e->setBytes(&K, sizeof(K), 5);
    e->setBytes(&lda, sizeof(lda), 6);
    e->setBytes(&ldb, sizeof(ldb), 7);
    e->setBytes(&ldc, sizeof(ldc), 8);
    e->dispatchThreadgroups(MTL::Size::Make((N + 63) / 64, (M + 63) / 64, 1),
                            MTL::Size::Make(16, 16, 1));
}

void embed8(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
            const Tensor& q, const Tensor& s, const Tensor& b,
            MTL::Buffer* tokens, MTL::Buffer* out, uint32_t T, uint32_t K) {
    auto p = m.pso("embed8");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(tokens, 0, 3);
    e->setBuffer(out, 0, 4);
    e->setBytes(&K, sizeof(K), 5);
    e->dispatchThreadgroups(MTL::Size::Make(T, 1, 1), MTL::Size::Make(256, 1, 1));
}

void conv_gate(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* bcx,
               MTL::Buffer* gated, uint32_t T, uint32_t H) {
    auto p = m.pso("conv_gate");
    e->setComputePipelineState(p);
    e->setBuffer(bcx, 0, 0);
    e->setBuffer(gated, 0, 1);
    e->setBytes(&T, sizeof(T), 2);
    e->setBytes(&H, sizeof(H), 3);
    uint64_t total = (uint64_t)T * H;
    e->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
}

void conv7(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* gated,
           MTL::Buffer* bcx, const Weights& w, const Tensor& kernel,
           MTL::Buffer* state, MTL::Buffer* state_out, MTL::Buffer* y,
           uint32_t T, uint32_t H) {
    auto p = m.pso("conv7");
    e->setComputePipelineState(p);
    e->setBuffer(gated, 0, 0);
    e->setBuffer(bcx, 0, 1);
    e->setBuffer(w.buffer(), kernel.begin, 2);
    e->setBuffer(state, 0, 3);
    e->setBuffer(state_out, 0, 4);
    e->setBuffer(y, 0, 5);
    e->setBytes(&T, sizeof(T), 6);
    e->setBytes(&H, sizeof(H), 7);
    uint64_t total = (uint64_t)T * H;
    e->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
}

void kv_append(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
               MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
               uint32_t pos, uint32_t max_seq, uint32_t kvh) {
    auto p = m.pso("kv_append");
    e->setComputePipelineState(p);
    e->setBuffer(k, 0, 0);
    e->setBuffer(v, 0, 1);
    e->setBuffer(kc, 0, 2);
    e->setBuffer(vc, 0, 3);
    e->setBytes(&pos, sizeof(pos), 4);
    e->setBytes(&max_seq, sizeof(max_seq), 5);
    e->setBytes(&kvh, sizeof(kvh), 6);
    uint64_t total = (uint64_t)kvh * 128;
    e->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
}

void rmsnorm_f32(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
                 const Weights& w, const Tensor& gamma, MTL::Buffer* y,
                 uint32_t rows, uint32_t n, float eps) {
    auto p = m.pso("rmsnorm_f32");
    e->setComputePipelineState(p);
    e->setBuffer(x, 0, 0);
    e->setBuffer(w.buffer(), gamma.begin, 1);
    e->setBuffer(y, 0, 2);
    e->setBytes(&n, sizeof(n), 3);
    e->setBytes(&eps, sizeof(eps), 4);
    e->dispatchThreadgroups(MTL::Size::Make(rows, 1, 1), MTL::Size::Make(256, 1, 1));
}

void to_bf16(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
             MTL::Buffer* y, uint32_t n) {
    auto p = m.pso("to_bf16");
    e->setComputePipelineState(p);
    e->setBuffer(x, 0, 0);
    e->setBuffer(y, 0, 1);
    e->setBytes(&n, sizeof(n), 2);
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
}

void gate_mul(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* attn,
              MTL::Buffer* gate, MTL::Buffer* out, uint32_t n) {
    auto p = m.pso("gate_mul");
    e->setComputePipelineState(p);
    e->setBuffer(attn, 0, 0);
    e->setBuffer(gate, 0, 1);
    e->setBuffer(out, 0, 2);
    e->setBytes(&n, sizeof(n), 3);
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
}

void rope(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
          MTL::Buffer* k, uint32_t pos, float theta, bool partial) {
    auto p = m.pso("rope");
    e->setComputePipelineState(p);
    e->setBuffer(q, 0, 0);
    e->setBuffer(k, 0, 1);
    e->setBytes(&pos, sizeof(pos), 2);
    e->setBytes(&theta, sizeof(theta), 3);
    uint32_t part = partial ? 1u : 0u;
    e->setBytes(&part, sizeof(part), 4);
    e->dispatchThreadgroups(MTL::Size::Make(20, 1, 1), MTL::Size::Make(128, 1, 1));
}

void attn_decode(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                 MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* out,
                 uint32_t pos, uint32_t max_seq, uint32_t kvh, uint32_t window) {
    auto p = m.pso("attn_decode");
    e->setComputePipelineState(p);
    e->setBuffer(q, 0, 0);
    e->setBuffer(kc, 0, 1);
    e->setBuffer(vc, 0, 2);
    e->setBuffer(out, 0, 3);
    e->setBytes(&pos, sizeof(pos), 4);
    e->setBytes(&max_seq, sizeof(max_seq), 5);
    e->setBytes(&kvh, sizeof(kvh), 6);
    e->setBytes(&window, sizeof(window), 7);
    e->dispatchThreadgroups(MTL::Size::Make(kvh, 1, 1), MTL::Size::Make(128, 1, 1));
}

void to_f32(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
            MTL::Buffer* y, uint32_t n) {
    auto p = m.pso("to_f32");
    e->setComputePipelineState(p);
    e->setBuffer(x, 0, 0);
    e->setBuffer(y, 0, 1);
    e->setBytes(&n, sizeof(n), 2);
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
}

void rope_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
            MTL::Buffer* k, uint32_t T, uint32_t start_pos, float theta, bool partial) {
    auto p = m.pso("rope_b");
    e->setComputePipelineState(p);
    e->setBuffer(q, 0, 0);
    e->setBuffer(k, 0, 1);
    e->setBytes(&start_pos, sizeof(start_pos), 2);
    e->setBytes(&theta, sizeof(theta), 3);
    uint32_t part = partial ? 1u : 0u;
    e->setBytes(&part, sizeof(part), 4);
    e->dispatchThreadgroups(MTL::Size::Make(T * 20, 1, 1), MTL::Size::Make(128, 1, 1));
}

void gate_mul_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* attn,
                MTL::Buffer* gate, MTL::Buffer* out, uint32_t total) {
    auto p = m.pso("gate_mul_b");
    e->setComputePipelineState(p);
    e->setBuffer(attn, 0, 0);
    e->setBuffer(gate, 0, 1);
    e->setBuffer(out, 0, 2);
    e->setBytes(&total, sizeof(total), 3);
    e->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
}

void kv_write_b(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* k,
                MTL::Buffer* v, MTL::Buffer* kc, MTL::Buffer* vc,
                uint32_t T, uint32_t start, uint32_t max_seq) {
    auto p = m.pso("kv_write_b");
    e->setComputePipelineState(p);
    e->setBuffer(k, 0, 0);
    e->setBuffer(v, 0, 1);
    e->setBuffer(kc, 0, 2);
    e->setBuffer(vc, 0, 3);
    e->setBytes(&T, sizeof(T), 4);
    e->setBytes(&start, sizeof(start), 5);
    e->setBytes(&max_seq, sizeof(max_seq), 6);
    uint64_t total = (uint64_t)T * 512;
    e->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
}

void softmax_prefill(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* scores,
                     MTL::Buffer* out, uint32_t T, uint32_t seq, uint32_t start_pos,
                     bool sliding) {
    auto p = m.pso("softmax_prefill");
    e->setComputePipelineState(p);
    e->setBuffer(scores, 0, 0);
    e->setBuffer(out, 0, 1);
    e->setBytes(&T, sizeof(T), 2);
    e->setBytes(&seq, sizeof(seq), 3);
    e->setBytes(&start_pos, sizeof(start_pos), 4);
    uint32_t sl = sliding ? 1u : 0u;
    e->setBytes(&sl, sizeof(sl), 5);
    e->dispatchThreadgroups(MTL::Size::Make(T, 1, 1), MTL::Size::Make(256, 1, 1));
}

void copy_row(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* src,
              MTL::Buffer* dst, uint32_t stride, uint32_t row, uint32_t n) {
    auto p = m.pso("copy_row");
    e->setComputePipelineState(p);
    e->setBuffer(src, 0, 0);
    e->setBuffer(dst, 0, 1);
    e->setBytes(&stride, sizeof(stride), 2);
    e->setBytes(&row, sizeof(row), 3);
    e->setBytes(&n, sizeof(n), 4);
    e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
}

void dequant_tile(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
                  const Tensor& q, const Tensor& s, const Tensor& b,
                  MTL::Buffer* out, uint32_t N, uint32_t K, uint32_t boff) {
    auto p = m.pso("dequant_tile");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(out, boff, 3);
    e->setBytes(&N, sizeof(N), 4);
    e->setBytes(&K, sizeof(K), 5);
    e->dispatchThreadgroups(MTL::Size::Make(K / 128, N / 64, 1), MTL::Size::Make(256, 1, 1));
}

void gemm_mpp(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* A,
              MTL::Buffer* B, MTL::Buffer* C, uint32_t M, uint32_t N, uint32_t K,
              uint32_t boff) {
    auto p = m.pso("gemm_mpp");
    e->setComputePipelineState(p);
    e->setBuffer(A, 0, 0);
    e->setBuffer(B, boff, 1);
    e->setBuffer(C, 0, 2);
    e->setBytes(&M, sizeof(M), 3);
    e->setBytes(&N, sizeof(N), 4);
    e->setBytes(&K, sizeof(K), 5);
    uint32_t lda = K, ldc = N;
    e->setBytes(&lda, sizeof(lda), 6);
    e->setBytes(&ldc, sizeof(ldc), 7);
    e->dispatchThreadgroups(MTL::Size::Make(N / 64, (M + 63) / 64, 1), MTL::Size::Make(128, 1, 1));
}

void gemv4x4(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
             const Tensor& q, const Tensor& s, const Tensor& b,
             MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K) {
    auto p = m.pso("gemv4x4");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(x, 0, 3);
    e->setBuffer(y, 0, 4);
    e->setBytes(&N, sizeof(N), 5);
    e->setBytes(&K, sizeof(K), 6);
    e->dispatchThreadgroups(MTL::Size::Make((N + 31) / 32, 1, 1), MTL::Size::Make(256, 1, 1));
}

void postnorm_add(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* x,
                  MTL::Buffer* src, const Weights& w, const Tensor& gamma,
                  MTL::Buffer* out, uint32_t rows, uint32_t n, float eps) {
    auto p = m.pso("post_norm_add");
    e->setComputePipelineState(p);
    e->setBuffer(x, 0, 0);
    e->setBuffer(src, 0, 1);
    e->setBuffer(w.buffer(), gamma.begin, 2);
    e->setBuffer(out, 0, 3);
    e->setBytes(&n, sizeof(n), 4);
    e->setBytes(&eps, sizeof(eps), 5);
    e->dispatchThreadgroups(MTL::Size::Make(rows, 1, 1), MTL::Size::Make(256, 1, 1));
}

void gemv4_raw(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
               MTL::Buffer* s, MTL::Buffer* b, MTL::Buffer* x, MTL::Buffer* y,
               uint32_t N, uint32_t K) {
    auto p = m.pso("gemv4");
    e->setComputePipelineState(p);
    e->setBuffer(q, 0, 0);
    e->setBuffer(s, 0, 1);
    e->setBuffer(b, 0, 2);
    e->setBuffer(x, 0, 3);
    e->setBuffer(y, 0, 4);
    e->setBytes(&N, sizeof(N), 5);
    e->setBytes(&K, sizeof(K), 6);
    e->dispatchThreadgroups(MTL::Size::Make((N + 15) / 16, 1, 1),
                            MTL::Size::Make(512, 1, 1));
}

void attn_decode_split(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* q,
                       MTL::Buffer* kc, MTL::Buffer* vc, MTL::Buffer* pacc,
                       MTL::Buffer* pm, MTL::Buffer* pl, uint32_t pos,
                       uint32_t max_seq, uint32_t kvh, uint32_t splits) {
    auto p = m.pso("attn_decode_split");
    e->setComputePipelineState(p);
    e->setBuffer(q, 0, 0);
    e->setBuffer(kc, 0, 1);
    e->setBuffer(vc, 0, 2);
    e->setBuffer(pacc, 0, 3);
    e->setBuffer(pm, 0, 4);
    e->setBuffer(pl, 0, 5);
    e->setBytes(&pos, sizeof(pos), 6);
    e->setBytes(&max_seq, sizeof(max_seq), 7);
    e->setBytes(&kvh, sizeof(kvh), 8);
    e->setBytes(&splits, sizeof(splits), 9);
    e->dispatchThreadgroups(MTL::Size::Make(kvh * splits, 1, 1), MTL::Size::Make(128, 1, 1));
}

void attn_combine(Metal& m, MTL::ComputeCommandEncoder* e, MTL::Buffer* pacc,
                  MTL::Buffer* pm, MTL::Buffer* pl, MTL::Buffer* out,
                  uint32_t splits) {
    auto p = m.pso("attn_combine");
    e->setComputePipelineState(p);
    e->setBuffer(pacc, 0, 0);
    e->setBuffer(pm, 0, 1);
    e->setBuffer(pl, 0, 2);
    e->setBuffer(out, 0, 3);
    e->setBytes(&splits, sizeof(splits), 4);
    e->dispatchThreadgroups(MTL::Size::Make(16, 1, 1), MTL::Size::Make(128, 1, 1));
}

void gemv4x2(Metal& m, MTL::ComputeCommandEncoder* e, const Weights& w,
             const Tensor& q, const Tensor& s, const Tensor& b,
             MTL::Buffer* x, MTL::Buffer* y, uint32_t N, uint32_t K) {
    auto p = m.pso("gemv4x2");
    e->setComputePipelineState(p);
    e->setBuffer(w.buffer(), q.begin, 0);
    e->setBuffer(w.buffer(), s.begin, 1);
    e->setBuffer(w.buffer(), b.begin, 2);
    e->setBuffer(x, 0, 3);
    e->setBuffer(y, 0, 4);
    e->setBytes(&N, sizeof(N), 5);
    e->setBytes(&K, sizeof(K), 6);
    e->dispatchThreadgroups(MTL::Size::Make((N + 15) / 16, 1, 1), MTL::Size::Make(256, 1, 1));
}

}  // namespace miori
