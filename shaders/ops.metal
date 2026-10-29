#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- 4-bit GEMV
// One warp per output row. q is [N, K/8] u32, s/b are [N, K/32] bf16.
kernel void gemv4(device const uint4* q [[buffer(0)]],
                  device const bfloat* s [[buffer(1)]],
                  device const bfloat* b [[buffer(2)]],
                  device const bfloat* x [[buffer(3)]],
                  device float* y [[buffer(4)]],
                  constant uint& N [[buffer(5)]],
                  constant uint& K [[buffer(6)]],
                  uint tg [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint row = tg * 16u + warp;
    if (row >= N) return;
    const uint ng = K / 32u;
    const uint qrow_words = K / 8u;
    device const uint4* qrow = q + (ulong)row * (qrow_words / 4u);
    device const bfloat* srow = s + (ulong)row * ng;
    device const bfloat* brow = b + (ulong)row * ng;
    float acc = 0.0f;
    for (uint g = lane; g < ng; g += 32u) {
        uint4 w = qrow[g];
        uint ww[4] = {w.x, w.y, w.z, w.w};
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        const uint base = g * 32u;
        #pragma unroll
        for (uint wi = 0; wi < 4u; ++wi) {
            const uint v = ww[wi];
            #pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                const float qv = float((v >> (4u * j)) & 0xFu);
                acc = fma(qv * sc + bi, float(x[base + wi * 8u + j]), acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- 8-bit GEMV
// q is [N, K/4] u32 (4 values per word), s/b are [N, K/32] bf16.
kernel void gemv8(device const uint4* q [[buffer(0)]],
                  device const bfloat* s [[buffer(1)]],
                  device const bfloat* b [[buffer(2)]],
                  device const bfloat* x [[buffer(3)]],
                  device float* y [[buffer(4)]],
                  constant uint& N [[buffer(5)]],
                  constant uint& K [[buffer(6)]],
                  uint tg [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint row = tg * 16u + warp;
    if (row >= N) return;
    const uint ng = K / 32u;
    device const uint4* qrow = q + (ulong)row * (K / 16u);  // K/4 u32 = K/16 uint4
    device const bfloat* srow = s + (ulong)row * ng;
    device const bfloat* brow = b + (ulong)row * ng;
    float acc = 0.0f;
    for (uint g = lane; g < ng; g += 32u) {
        uint4 w0 = qrow[2u * g];
        uint4 w1 = qrow[2u * g + 1u];
        uint ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        const uint base = g * 32u;
        #pragma unroll
        for (uint wi = 0; wi < 8u; ++wi) {
            const uint v = ww[wi];
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float qv = float((v >> (8u * j)) & 0xFFu);
                acc = fma(qv * sc + bi, float(x[base + wi * 4u + j]), acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- RMSNorm
// One threadgroup per row, 256 threads.
kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const bfloat* w [[buffer(1)]],
                    device bfloat* y [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    constant float& eps [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const ulong ro = (ulong)tg * n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        const float v = x[ro + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) part[warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint i = 0; i < 8u; ++i) tot += part[i];
        part[0] = rsqrt(tot / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = part[0];
    for (uint i = tid; i < n; i += 256u) {
        y[ro + i] = bfloat(x[ro + i] * inv * float(w[i]));
    }
}

// ---------------------------------------------------------------- residual add
kernel void add(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* out [[buffer(2)]],
                constant uint& n [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

// y = x + norm(op) with post-norm already applied; the next RMSNorm is a separate kernel.

// ---------------------------------------------------------------- SiLU * up
kernel void silu_mul(device const float* gate [[buffer(0)]],
                     device const float* up [[buffer(1)]],
                     device bfloat* out [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) {
        const float g = gate[i];
        const float s = g / (1.0f + exp(-g));
        out[i] = bfloat(s * up[i]);
    }
}

// ---------------------------------------------------------------- dequant to bf16 (row-major)
kernel void dequant4(device const uint4* q [[buffer(0)]],
                     device const bfloat* s [[buffer(1)]],
                     device const bfloat* b [[buffer(2)]],
                     device bfloat* out [[buffer(3)]],
                     constant uint& N [[buffer(4)]],
                     constant uint& K [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    const uint ng = K / 32u;
    const uint total = N * ng;
    if (gid >= total) return;
    const uint row = gid / ng, g = gid % ng;
    device const uint4* qrow = q + (ulong)row * (K / 32u);
    const uint4 w = qrow[g];
    const uint ww[4] = {w.x, w.y, w.z, w.w};
    const float sc = float(s[(ulong)row * ng + g]);
    const float bi = float(b[(ulong)row * ng + g]);
    const ulong base = (ulong)row * K + (ulong)g * 32u;
    #pragma unroll
    for (uint wi = 0; wi < 4u; ++wi) {
        const uint v = ww[wi];
        #pragma unroll
        for (uint j = 0; j < 8u; ++j)
            out[base + wi * 8u + j] = bfloat(float((v >> (4u * j)) & 0xFu) * sc + bi);
    }
}

// ---------------------------------------------------------------- embedding (8-bit)
// One threadgroup per token; 256 threads; dequant one row.
kernel void embed8(device const uint4* q [[buffer(0)]],
                   device const bfloat* s [[buffer(1)]],
                   device const bfloat* b [[buffer(2)]],
                   device const uint* tokens [[buffer(3)]],
                   device bfloat* out [[buffer(4)]],
                   constant uint& K [[buffer(5)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tokens[tg];
    const uint ng = K / 32u;
    device const uint4* qrow = q + (ulong)tok * (K / 16u);
    device const bfloat* srow = s + (ulong)tok * ng;
    device const bfloat* brow = b + (ulong)tok * ng;
    for (uint g = tid; g < ng; g += 256u) {
        const uint4 w0 = qrow[2u * g];
        const uint4 w1 = qrow[2u * g + 1u];
        const uint ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        for (uint wi = 0; wi < 8u; ++wi) {
            const uint v = ww[wi];
            for (uint j = 0; j < 4u; ++j) {
                const uint k = g * 32u + wi * 4u + j;
                out[(ulong)tg * K + k] = bfloat(float((v >> (8u * j)) & 0xFFu) * sc + bi);
            }
        }
    }
}

// ---------------------------------------------------------------- naive bf16 GEMM
// C[M,N] = A[M,K] * B[N,K]^T, all row-major; 32x32 tile, 256 threads, 2x2 per thread.
kernel void gemm_naive(device const bfloat* A [[buffer(0)]],
                       device const bfloat* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       constant uint& M [[buffer(3)]],
                       constant uint& N [[buffer(4)]],
                       constant uint& K [[buffer(5)]],
                       constant uint& lda [[buffer(6)]],
                       constant uint& ldb [[buffer(7)]],
                       constant uint& ldc [[buffer(8)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float As[64][17];
    threadgroup float Bs[64][17];
    const uint tx = tid.x, ty = tid.y;
    const uint m0 = tgid.y * 64u, n0 = tgid.x * 64u;
    float acc[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    for (uint k0 = 0; k0 < K; k0 += 16u) {
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            const uint m = m0 + ty * 4u + r;
            const uint k = k0 + tx;
            As[ty * 4u + r][tx] = (m < M && k < K) ? float(A[(ulong)m * lda + k]) : 0.0f;
            const uint n = n0 + ty * 4u + r;
            Bs[ty * 4u + r][tx] = (n < N && k < K) ? float(B[(ulong)n * ldb + k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < 16u; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (uint r = 0; r < 4u; ++r) a[r] = As[ty * 4u + r][kk];
            #pragma unroll
            for (uint c = 0; c < 4u; ++c) b[c] = Bs[tx * 4u + c][kk];
            #pragma unroll
            for (uint r = 0; r < 4u; ++r)
                #pragma unroll
                for (uint c = 0; c < 4u; ++c)
                    acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        #pragma unroll
        for (uint c = 0; c < 4u; ++c) {
            const uint m = m0 + ty * 4u + r;
            const uint n = n0 + tx * 4u + c;
            if (m < M && n < N) C[(ulong)m * ldc + n] = acc[r][c];
        }
    }
}

// ---------------------------------------------------------------- conv gate / conv
// gated[t,c] = BCx[t,c] * BCx[t,2H+c]
kernel void conv_gate(device const float* bcx [[buffer(0)]],
                      device float* gated [[buffer(1)]],
                      constant uint& T [[buffer(2)]],
                      constant uint& H [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    const uint total = T * H;
    if (i >= total) return;
    const uint t = i / H, c = i % H;
    gated[(ulong)t * H + c] = bcx[(ulong)t * 3u * H + c] * bcx[(ulong)t * 3u * H + 2u * H + c];
}

// y[t,c] = C[t,c] * sum_{k=0..6} w[c,k] * gated[t+k-6,c]
// state[6][H] holds the 6 inputs before t=0.
kernel void conv7(device const float* gated [[buffer(0)]],
                  device const float* bcx [[buffer(1)]],
                  device const bfloat* w [[buffer(2)]],      // [H,7]
                  device const float* state [[buffer(3)]],    // [6,H]
                  device float* state_out [[buffer(4)]],      // [6,H]
                  device float* y [[buffer(5)]],
                  constant uint& T [[buffer(6)]],
                  constant uint& H [[buffer(7)]],
                  uint i [[thread_position_in_grid]]) {
    const uint total = T * H;
    if (i >= total) return;
    const uint t = i / H, c = i % H;
    float acc = 0.0f;
    for (uint k = 0; k < 7u; ++k) {
        const int pos = (int)t + (int)k - 6;
        float v;
        if (pos < 0) v = state[(uint)(pos + 6) * H + c];
        else v = gated[(ulong)pos * H + c];
        acc = fma(v, float(w[c * 7u + k]), acc);
    }
    y[(ulong)t * H + c] = bcx[(ulong)t * 3u * H + H + c] * acc;

    // Shift the 6-token window: last 6 of (old_state ++ gated[0..T-1]).
    if (t == 0) {
        for (uint k = 0; k < 6u; ++k) {
            const int idx = (int)T - 6 + (int)k;
            float v;
            if (idx < 0) v = state[(uint)(idx + 6) * H + c];
            else v = gated[(ulong)idx * H + c];
            state_out[k * H + c] = v;
        }
    }
}

// ---------------------------------------------------------------- KV append
// K cache: [max_seq][kv_heads][128]; Vt cache: [kv_heads][128][max_seq].
kernel void kv_append(device const bfloat* k [[buffer(0)]],
                      device const float* v [[buffer(1)]],
                      device bfloat* kc [[buffer(2)]],
                      device bfloat* vc [[buffer(3)]],
                      constant uint& pos [[buffer(4)]],
                      constant uint& max_seq [[buffer(5)]],
                      constant uint& kvh [[buffer(6)]],
                      uint i [[thread_position_in_grid]]) {
    const uint total = kvh * 128u;
    if (i >= total) return;
    const uint h = i / 128u, d = i % 128u;
    kc[((ulong)h * max_seq + pos) * 128u + d] = k[h * 128u + d];
    vc[(h * 128u + d) * max_seq + pos] = bfloat(v[h * 128u + d]);
}

// ---------------------------------------------------------------- RMSNorm (float out)
kernel void rmsnorm_f32(device const float* x [[buffer(0)]],
                        device const bfloat* w [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        constant float& eps [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const ulong ro = (ulong)tg * n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += 256u) { const float v = x[ro + i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0u) part[warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint i = 0; i < 8u; ++i) tot += part[i];
        part[0] = rsqrt(tot / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = part[0];
    for (uint i = tid; i < n; i += 256u) y[ro + i] = x[ro + i] * inv * float(w[i]);
}

// ---------------------------------------------------------------- f32 -> bf16
kernel void to_bf16(device const float* x [[buffer(0)]],
                    device bfloat* y [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = bfloat(x[i]);
}

// ---------------------------------------------------------------- attention gate
kernel void gate_mul(device const float* attn [[buffer(0)]],
                     device const float* gate [[buffer(1)]],
                     device bfloat* out [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) {
        const uint h = i / 128u;
        out[i] = bfloat(attn[i] / (1.0f + exp(-gate[h])));
    }
}

// ---------------------------------------------------------------- RoPE (q + k, in place)
// q: [16][128], k: [4][128]. tg 0..15 = q heads, 16..19 = k heads.
kernel void rope(device bfloat* q [[buffer(0)]],
                 device bfloat* k [[buffer(1)]],
                 constant uint& pos [[buffer(2)]],
                 constant float& theta [[buffer(3)]],
                 constant uint& partial [[buffer(4)]],
                 uint tg [[threadgroup_position_in_grid]],
                 uint i [[thread_index_in_threadgroup]]) {
    if (i >= 64u) return;
    if (partial && i >= 32u) return;
    const bool is_k = tg >= 16u;
    const uint head = is_k ? tg - 16u : tg;
    device bfloat* base = is_k ? k : q;
    const float inv = 1.0f / pow(theta, float(2u * i) / 128.0f);
    const float ang = float(pos) * inv;
    const float c = cos(ang), s = sin(ang);
    const uint a = head * 128u + i, b = a + 64u;
    const float x0 = float(base[a]), x1 = float(base[b]);
    base[a] = bfloat(x0 * c - x1 * s);
    base[b] = bfloat(x1 * c + x0 * s);
}

// ---------------------------------------------------------------- decode attention
// One threadgroup per KV head, 4 warps = 4 query heads. q/k/v bf16.
kernel void attn_decode(device const bfloat* q [[buffer(0)]],
                        device const bfloat* kc [[buffer(1)]],
                        device const bfloat* vc [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        constant uint& pos [[buffer(4)]],
                        constant uint& max_seq [[buffer(5)]],
                        constant uint& kvh [[buffer(6)]],
                        constant uint& window [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg;
    const uint warp = tid / 32u, lane = tid % 32u;
    const uint qh = h * 4u + warp;
    const uint d0 = lane * 4u;
    float q0 = float(q[qh * 128u + d0 + 0u]);
    float q1 = float(q[qh * 128u + d0 + 1u]);
    float q2 = float(q[qh * 128u + d0 + 2u]);
    float q3 = float(q[qh * 128u + d0 + 3u]);
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint keys = pos + 1u;
    const uint start = (window && keys > window) ? keys - window : 0u;
    for (uint t = start; t < keys; ++t) {
        const ulong kb = ((ulong)h * max_seq + t) * 128u + d0;
        float s = q0 * float(kc[kb + 0u]) + q1 * float(kc[kb + 1u]) +
                  q2 * float(kc[kb + 2u]) + q3 * float(kc[kb + 3u]);
        s = simd_sum(s) * 0.08838834764831845f;
        const float nm = max(m, s);
        const float alpha = exp(m - nm);
        const float p = exp(s - nm);
        l = l * alpha + p;
        const ulong vb = ((ulong)h * 128u + d0) * max_seq + t;
        a0 = a0 * alpha + p * float(vc[vb + 0ul * max_seq]);
        a1 = a1 * alpha + p * float(vc[vb + 1ul * max_seq]);
        a2 = a2 * alpha + p * float(vc[vb + 2ul * max_seq]);
        a3 = a3 * alpha + p * float(vc[vb + 3ul * max_seq]);
        m = nm;
    }
    const float inv = 1.0f / l;
    out[qh * 128u + d0 + 0u] = a0 * inv;
    out[qh * 128u + d0 + 1u] = a1 * inv;
    out[qh * 128u + d0 + 2u] = a2 * inv;
    out[qh * 128u + d0 + 3u] = a3 * inv;
}

// ---------------------------------------------------------------- bf16 -> f32
kernel void to_f32(device const bfloat* x [[buffer(0)]],
                   device float* y [[buffer(1)]],
                   constant uint& n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = float(x[i]);
}

// ---------------------------------------------------------------- RoPE for a prefill chunk
// q: [T][16][128], k: [T][4][128]. tg = t*20 + hd.
kernel void rope_b(device bfloat* q [[buffer(0)]],
                   device bfloat* k [[buffer(1)]],
                   constant uint& start_pos [[buffer(2)]],
                   constant float& theta [[buffer(3)]],
                   constant uint& partial [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint i [[thread_index_in_threadgroup]]) {
    if (i >= 64u) return;
    if (partial && i >= 32u) return;
    const uint t = tg / 20u, hd = tg % 20u;
    const bool is_k = hd >= 16u;
    const uint head = is_k ? hd - 16u : hd;
    device bfloat* base = is_k ? k : q;
    const uint stride = is_k ? 4u : 16u;
    const uint pos = start_pos + t;
    const float inv = 1.0f / pow(theta, float(2u * i) / 128.0f);
    const float ang = float(pos) * inv;
    const float c = cos(ang), s = sin(ang);
    const uint a = (t * stride + head) * 128u + i, b = a + 64u;
    const float x0 = float(base[a]), x1 = float(base[b]);
    base[a] = bfloat(x0 * c - x1 * s);
    base[b] = bfloat(x1 * c + x0 * s);
}

// ---------------------------------------------------------------- attention gate (batched)
kernel void gate_mul_b(device const float* attn [[buffer(0)]],
                       device const float* gate [[buffer(1)]],
                       device bfloat* out [[buffer(2)]],
                       constant uint& total [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < total) {
        const uint t = i / 2048u, h = (i % 2048u) / 128u;
        out[i] = bfloat(attn[i] / (1.0f + exp(-gate[t * 16u + h])));
    }
}

// ---------------------------------------------------------------- KV write (prefill chunk)
kernel void kv_write_b(device const bfloat* k [[buffer(0)]],
                       device const float* v [[buffer(1)]],
                       device bfloat* kc [[buffer(2)]],
                       device bfloat* vc [[buffer(3)]],
                       constant uint& T [[buffer(4)]],
                       constant uint& start [[buffer(5)]],
                       constant uint& max_seq [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    const uint total = T * 4u * 128u;
    if (i >= total) return;
    const uint t = i / 512u, r = i % 512u;
    const uint h = r / 128u, d = r % 128u;
    const uint pos = start + t;
    kc[((ulong)h * max_seq + pos) * 128u + d] = k[(ulong)t * 512u + r];
    vc[((ulong)h * 128u + d) * max_seq + pos] = bfloat(v[(ulong)t * 512u + r]);
}

// ---------------------------------------------------------------- causal/sliding softmax
kernel void softmax_prefill(device float* scores [[buffer(0)]],
                            device bfloat* out [[buffer(1)]],
                            constant uint& T [[buffer(2)]],
                            constant uint& seq [[buffer(3)]],
                            constant uint& start_pos [[buffer(4)]],
                            constant uint& sliding [[buffer(5)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const uint pos = start_pos + row;
    const float scale = 0.08838834764831845f;
    float mx = -INFINITY;
    for (uint s = tid; s < seq; s += 256u) {
        const bool ok = (s <= pos) && (!sliding || (s + 256u > pos));
        if (ok) mx = max(mx, scores[(ulong)row * seq + s] * scale);
    }
    mx = simd_max(mx);
    if (lane == 0u) part[warp] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = part[0];
        for (uint i = 1; i < 8u; ++i) m = max(m, part[i]);
        part[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = part[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint s = tid; s < seq; s += 256u) {
        const bool ok = (s <= pos) && (!sliding || (s + 256u > pos));
        const float e = ok ? exp(scores[(ulong)row * seq + s] * scale - mx) : 0.0f;
        sum += e;
        scores[(ulong)row * seq + s] = e;
    }
    sum = simd_sum(sum);
    if (lane == 0u) part[warp] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float s = part[0];
        for (uint i = 1; i < 8u; ++i) s += part[i];
        part[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = part[0];
    for (uint s = tid; s < seq; s += 256u)
        out[(ulong)row * seq + s] = bfloat(scores[(ulong)row * seq + s] * inv);
}

// ---------------------------------------------------------------- copy one bf16 row
kernel void copy_row(device const bfloat* src [[buffer(0)]],
                     device bfloat* dst [[buffer(1)]],
                     constant uint& stride [[buffer(2)]],
                     constant uint& row [[buffer(3)]],
                     constant uint& n [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[(ulong)row * stride + i];
}

// 4-bit GEMV, 4 rows per warp (8 lanes per row). More ILP / wider coalescing.
kernel void gemv4x4(device const uint4* q [[buffer(0)]],
                    device const bfloat* s [[buffer(1)]],
                    device const bfloat* b [[buffer(2)]],
                    device const bfloat* x [[buffer(3)]],
                    device float* y [[buffer(4)]],
                    constant uint& N [[buffer(5)]],
                    constant uint& K [[buffer(6)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint grp = lane / 8u;
    const uint li = lane % 8u;
    const uint row = tg * 32u + warp * 4u + grp;
    if (row >= N) return;
    const uint ng = K / 32u;
    device const uint4* qrow = q + (ulong)row * (K / 32u);
    device const bfloat* srow = s + (ulong)row * ng;
    device const bfloat* brow = b + (ulong)row * ng;
    float acc = 0.0f;
    for (uint g = li; g < ng; g += 8u) {
        uint4 w = qrow[g];
        uint ww[4] = {w.x, w.y, w.z, w.w};
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        const uint base = g * 32u;
        #pragma unroll
        for (uint wi = 0; wi < 4u; ++wi) {
            const uint v = ww[wi];
            #pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                const float qv = float((v >> (4u * j)) & 0xFu);
                acc = fma(qv * sc + bi, float(x[base + wi * 8u + j]), acc);
            }
        }
    }
    acc += simd_shuffle_down(acc, 1);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 4);
    if (li == 0u) y[row] = acc;
}

// ---------------------------------------------------------------- post-norm + residual add
kernel void post_norm_add(device const float* x [[buffer(0)]],
                          device const float* src [[buffer(1)]],
                          device const bfloat* w [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          constant uint& n [[buffer(4)]],
                          constant float& eps [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const ulong ro = (ulong)tg * n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += 256u) { const float v = src[ro + i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0u) part[warp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint i = 0; i < 8u; ++i) tot += part[i];
        part[0] = rsqrt(tot / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = part[0];
    for (uint i = tid; i < n; i += 256u)
        out[ro + i] = x[ro + i] + src[ro + i] * inv * float(w[i]);
}
