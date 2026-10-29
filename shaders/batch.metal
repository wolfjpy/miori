// C2 — batch decode kernels.
//
// Layouts mirror the single-sequence kernels in ops.metal, with a leading slot
// dimension:
//   K cache: [B][kv_head][max_seq][128] bf16
//   V cache: [B][kv_head][128][max_seq] bf16
//   q:       [B][2048] bf16, k/v: [B][512] (v is f32 before the append)
//   conv state: [B][2][6][H] f32 (double buffered, `cur` selects the live half)
// Positions are per slot; exactly one token per slot is appended per step.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- RoPE (batch)
// q: [B][16][128], k: [B][4][128]. tg = slot*20 + head_or_khead.
kernel void rope_batch(device bfloat* q [[buffer(0)]],
                       device bfloat* k [[buffer(1)]],
                       device const uint* positions [[buffer(2)]],
                       constant float& theta [[buffer(3)]],
                       constant uint& partial [[buffer(4)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint i [[thread_index_in_threadgroup]]) {
    if (i >= 64u) return;
    if (partial && i >= 32u) return;
    const uint slot = tg / 20u, hd = tg % 20u;
    const bool is_k = hd >= 16u;
    const uint head = is_k ? hd - 16u : hd;
    device bfloat* base = is_k ? k : q;
    const uint stride = is_k ? 4u : 16u;
    const uint pos = positions[slot];
    const float inv = 1.0f / pow(theta, float(2u * i) / 128.0f);
    const float ang = float(pos) * inv;
    const float c = cos(ang), s = sin(ang);
    const uint a = (slot * stride + head) * 128u + i, b = a + 64u;
    const float x0 = float(base[a]), x1 = float(base[b]);
    base[a] = bfloat(x0 * c - x1 * s);
    base[b] = bfloat(x1 * c + x0 * s);
}

// ---------------------------------------------------------------- KV append (batch)
// One token per slot at positions[slot].
kernel void kv_append_b(device const bfloat* k [[buffer(0)]],
                        device const float* v [[buffer(1)]],
                        device bfloat* kc [[buffer(2)]],
                        device bfloat* vc [[buffer(3)]],
                        device const uint* positions [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        constant uint& max_seq [[buffer(6)]],
                        constant uint& kvh [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
    const uint total = B * kvh * 128u;
    if (i >= total) return;
    const uint slot = i / (kvh * 128u);
    const uint r = i % (kvh * 128u);
    const uint h = r / 128u, d = r % 128u;
    const uint pos = positions[slot];
    const ulong src = (ulong)slot * 512ul + r;
    kc[(((ulong)slot * kvh + h) * max_seq + pos) * 128ul + d] = k[src];
    vc[(((ulong)slot * kvh + h) * 128ul + d) * max_seq + pos] = bfloat(v[src]);
}

// ---------------------------------------------------------------- decode attention (batch)
// grid = B * kvh threadgroups; 4 warps per threadgroup = 4 query heads.
kernel void attn_decode_b(device const bfloat* q [[buffer(0)]],
                          device const bfloat* kc [[buffer(1)]],
                          device const bfloat* vc [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          device const uint* positions [[buffer(4)]],
                          constant uint& max_seq [[buffer(5)]],
                          constant uint& kvh [[buffer(6)]],
                          constant uint& window [[buffer(7)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const uint slot = tg / kvh, h = tg % kvh;
    const uint warp = tid / 32u, lane = tid % 32u;
    const uint qh = h * 4u + warp;
    const uint d0 = lane * 4u;
    device const bfloat* qs = q + (ulong)slot * 2048ul;
    device const bfloat* ks = kc + ((ulong)slot * kvh + h) * (ulong)max_seq * 128ul;
    device const bfloat* vs = vc + ((ulong)slot * kvh + h) * 128ul * (ulong)max_seq;
    const uint pos = positions[slot];
    const uint keys = pos + 1u;
    const uint start = (window && keys > window) ? keys - window : 0u;
    float q0 = float(qs[qh * 128u + d0 + 0u]);
    float q1 = float(qs[qh * 128u + d0 + 1u]);
    float q2 = float(qs[qh * 128u + d0 + 2u]);
    float q3 = float(qs[qh * 128u + d0 + 3u]);
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint t = start; t < keys; ++t) {
        const ulong kb = (ulong)t * 128ul + d0;
        float s = q0 * float(ks[kb + 0u]) + q1 * float(ks[kb + 1u]) +
                  q2 * float(ks[kb + 2u]) + q3 * float(ks[kb + 3u]);
        s = simd_sum(s) * 0.08838834764831845f;
        const float nm = max(m, s);
        const float alpha = exp(m - nm);
        const float p = exp(s - nm);
        l = l * alpha + p;
        const ulong vb = (ulong)d0 * (ulong)max_seq + t;
        a0 = a0 * alpha + p * float(vs[vb + 0ul * max_seq]);
        a1 = a1 * alpha + p * float(vs[vb + 1ul * max_seq]);
        a2 = a2 * alpha + p * float(vs[vb + 2ul * max_seq]);
        a3 = a3 * alpha + p * float(vs[vb + 3ul * max_seq]);
        m = nm;
    }
    const float inv = 1.0f / l;
    device float* o = out + (ulong)slot * 2048ul + qh * 128u + d0;
    o[0u] = a0 * inv;
    o[1u] = a1 * inv;
    o[2u] = a2 * inv;
    o[3u] = a3 * inv;
}

// ---------------------------------------------------------------- conv7 (batch)
// T = 1 per slot: y[s,c] = C[s,c] * sum_{k<6} state[k,c]*w[c,k] + gated[s,c]*w[c,6]
// The 6-token window is shifted into the other half of the double buffer.
kernel void conv7_b(device const float* gated [[buffer(0)]],
                    device const float* bcx [[buffer(1)]],
                    device const bfloat* w [[buffer(2)]],       // [H,7]
                    device float* state [[buffer(3)]],          // [B][2][6][H]
                    device bfloat* y [[buffer(4)]],             // [B][H]
                    constant uint& cur [[buffer(5)]],
                    constant uint& B [[buffer(6)]],
                    constant uint& H [[buffer(7)]],
                    uint i [[thread_position_in_grid]]) {
    const uint total = B * H;
    if (i >= total) return;
    const uint slot = i / H, c = i % H;
    device const float* st = state + ((ulong)slot * 2ul + cur) * 6ul * H;
    device float* sto = state + ((ulong)slot * 2ul + (1u - cur)) * 6ul * H;
    float acc = 0.0f;
    for (uint k = 0; k < 6u; ++k) acc = fma(st[k * H + c], float(w[c * 7u + k]), acc);
    acc = fma(gated[(ulong)slot * H + c], float(w[c * 7u + 6u]), acc);
    y[(ulong)slot * H + c] = bfloat(bcx[(ulong)slot * 3u * H + H + c] * acc);
    for (uint k = 0; k < 5u; ++k) sto[k * H + c] = st[(k + 1u) * H + c];
    sto[5u * H + c] = gated[(ulong)slot * H + c];
}


// ---------------------------------------------------------------- 4-bit GEMV (batch)
// One warp per output row; the weight row is read once and dotted with all B
// input rows at once. Accumulation order per row matches gemv4, so each row is
// numerically identical to the single-sequence kernel.
kernel void gemv4_b(device const uint4* q [[buffer(0)]],
                    device const bfloat* s [[buffer(1)]],
                    device const bfloat* b [[buffer(2)]],
                    device const bfloat* x [[buffer(3)]],   // [B][K]
                    device float* y [[buffer(4)]],          // [B][N]
                    constant uint& N [[buffer(5)]],
                    constant uint& K [[buffer(6)]],
                    constant uint& B [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint warp [[simdgroup_index_in_threadgroup]]) {
    if (B > 4u) return;
    const uint row = tg * 16u + warp;
    if (row >= N) return;
    const uint ng = K / 32u;
    device const uint4* qrow = q + (ulong)row * (K / 32u);
    device const bfloat* srow = s + (ulong)row * ng;
    device const bfloat* brow = b + (ulong)row * ng;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
                const float qv = float((v >> (4u * j)) & 0xFu) * sc + bi;
                const uint k = base + wi * 8u + j;
                #pragma unroll
                for (uint r = 0; r < 4u; ++r)
                    acc[r] = fma(qv, float(x[(ulong)r * K + k]), acc[r]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0u) y[(ulong)r * N + row] = v;
    }
}

// ---------------------------------------------------------------- 8-bit GEMV (batch)
// Batched lm_head: one pass over the 262 MB embedding matrix for all B rows.
kernel void gemv8_b(device const uint4* q [[buffer(0)]],
                    device const bfloat* s [[buffer(1)]],
                    device const bfloat* b [[buffer(2)]],
                    device const bfloat* x [[buffer(3)]],   // [B][K]
                    device float* y [[buffer(4)]],          // [B][N]
                    constant uint& N [[buffer(5)]],
                    constant uint& K [[buffer(6)]],
                    constant uint& B [[buffer(7)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint warp [[simdgroup_index_in_threadgroup]]) {
    if (B > 4u) return;
    const uint row = tg * 16u + warp;
    if (row >= N) return;
    const uint ng = K / 32u;
    device const uint4* qrow = q + (ulong)row * (K / 16u);
    device const bfloat* srow = s + (ulong)row * ng;
    device const bfloat* brow = b + (ulong)row * ng;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
                const float qv = float((v >> (8u * j)) & 0xFFu) * sc + bi;
                const uint k = base + wi * 4u + j;
                #pragma unroll
                for (uint r = 0; r < 4u; ++r)
                    acc[r] = fma(qv, float(x[(ulong)r * K + k]), acc[r]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0u) y[(ulong)r * N + row] = v;
    }
}

// ---------------------------------------------------------------- packed prefill
// The kernels below implement packed multi-request prefill (BatchEngine::add_batch):
// several prompts are concatenated into one [T][...] activation buffer, so the
// 24 layers' GEMMs run once for all requests. Only the request-boundary work
// (KV writes, attention, conv state) stays per request.

// KV write for a packed [T][512] chunk. Row r belongs to request slot slots[r]
// and sits at position positions[r] (both restart at 0 for every request).
kernel void kv_write_packed(device const bfloat* k [[buffer(0)]],
                            device const float* v [[buffer(1)]],
                            device bfloat* kc [[buffer(2)]],
                            device bfloat* vc [[buffer(3)]],
                            device const uint* slots [[buffer(4)]],
                            device const uint* positions [[buffer(5)]],
                            constant uint& max_seq [[buffer(6)]],
                            constant uint& kvh [[buffer(7)]],
                            uint i [[thread_position_in_grid]]) {
    const uint row = i / 512u, r = i % 512u;
    const uint h = r / 128u, d = r % 128u;
    const uint slot = slots[row], pos = positions[row];
    kc[(((ulong)slot * kvh + h) * max_seq + pos) * 128ul + d] = k[(ulong)row * 512ul + r];
    vc[(((ulong)slot * kvh + h) * 128ul + d) * max_seq + pos] =
        bfloat(v[(ulong)row * 512ul + r]);
}

// Gather the last row of every packed request out of a [T][K] bf16 buffer:
// dst[r][i] = src[rows[r]][i]. One threadgroup per request.
kernel void gather_rows(device const bfloat* src [[buffer(0)]],
                        device bfloat* dst [[buffer(1)]],
                        device const uint* rows [[buffer(2)]],
                        constant uint& K [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    device const bfloat* s = src + (ulong)rows[tg] * K;
    device bfloat* d = dst + (ulong)tg * K;
    for (uint i = tid; i < K; i += 256u) d[i] = s[i];
}

// ============================================================ int8 KV (batch)
// Same layout as the single-sequence int8 cache with a leading slot dimension:
//   K: [B][kvh][max_seq][128] int8, ks: [B][kvh][max_seq] bf16
//   V: [B][kvh][128][max_seq] int8, vs: [B][kvh][max_seq] bf16
// Scales are per (slot, kv_head, token); no zero point.

static inline float kvb_absmax(threadgroup float* sh, float v, uint lane, uint warp) {
    v = simd_max(v);
    if (lane == 0u) sh[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return max(max(sh[0], sh[1]), max(sh[2], sh[3]));
}

// One token per slot at positions[slot]; grid = B*kvh threadgroups of 128 threads.
kernel void kv_append_b_i8(device const bfloat* k [[buffer(0)]],
                           device const float* v [[buffer(1)]],
                           device char* kc [[buffer(2)]],
                           device char* vc [[buffer(3)]],
                           device bfloat* ks [[buffer(4)]],
                           device bfloat* vs [[buffer(5)]],
                           device const uint* positions [[buffer(6)]],
                           constant uint& max_seq [[buffer(7)]],
                           constant uint& kvh [[buffer(8)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tg / kvh, h = tg % kvh;
    const uint d = tid;
    const uint pos = positions[slot];
    const ulong src = (ulong)slot * 512ul + h * 128u + d;
    const float kv = float(k[src]);
    const float vv = v[src];
    threadgroup float mk[4], mv[4];
    const float kmax = kvb_absmax(mk, fabs(kv), lane, warp);
    const float vmax = kvb_absmax(mv, fabs(vv), lane, warp);
    const float kscale = kmax > 0.0f ? kmax / 127.0f : 0.0f;
    const float vscale = vmax > 0.0f ? vmax / 127.0f : 0.0f;
    const int kq = kscale > 0.0f ? (int)rint(clamp(kv / kscale, -127.0f, 127.0f)) : 0;
    const int vq = vscale > 0.0f ? (int)rint(clamp(vv / vscale, -127.0f, 127.0f)) : 0;
    const ulong kbase = ((ulong)slot * kvh + h) * max_seq;
    kc[(kbase + pos) * 128ul + d] = (char)kq;
    vc[(((ulong)slot * kvh + h) * 128ul + d) * max_seq + pos] = (char)vq;
    if (tid == 0u) {
        ks[kbase + pos] = bfloat(kscale);
        vs[kbase + pos] = bfloat(vscale);
    }
}

// Packed [T][512] prefill rows -> per-request slot/position; grid = T*kvh.
kernel void kv_write_packed_i8(device const bfloat* k [[buffer(0)]],
                               device const float* v [[buffer(1)]],
                               device char* kc [[buffer(2)]],
                               device char* vc [[buffer(3)]],
                               device bfloat* ks [[buffer(4)]],
                               device bfloat* vs [[buffer(5)]],
                               device const uint* slots [[buffer(6)]],
                               device const uint* positions [[buffer(7)]],
                               constant uint& max_seq [[buffer(8)]],
                               constant uint& kvh [[buffer(9)]],
                               uint tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint t = tg / kvh, h = tg % kvh;
    const uint d = tid;
    const uint slot = slots[t], pos = positions[t];
    const ulong src = (ulong)t * 512ul + h * 128u + d;
    const float kv = float(k[src]);
    const float vv = v[src];
    threadgroup float mk[4], mv[4];
    const float kmax = kvb_absmax(mk, fabs(kv), lane, warp);
    const float vmax = kvb_absmax(mv, fabs(vv), lane, warp);
    const float kscale = kmax > 0.0f ? kmax / 127.0f : 0.0f;
    const float vscale = vmax > 0.0f ? vmax / 127.0f : 0.0f;
    const int kq = kscale > 0.0f ? (int)rint(clamp(kv / kscale, -127.0f, 127.0f)) : 0;
    const int vq = vscale > 0.0f ? (int)rint(clamp(vv / vscale, -127.0f, 127.0f)) : 0;
    const ulong kbase = ((ulong)slot * kvh + h) * max_seq;
    kc[(kbase + pos) * 128ul + d] = (char)kq;
    vc[(((ulong)slot * kvh + h) * 128ul + d) * max_seq + pos] = (char)vq;
    if (tid == 0u) {
        ks[kbase + pos] = bfloat(kscale);
        vs[kbase + pos] = bfloat(vscale);
    }
}

// int8 batch decode attention: grid = B*kvh threadgroups, 4 warps = 4 q heads.
kernel void attn_decode_b_i8(device const bfloat* q [[buffer(0)]],
                             device const char* kc [[buffer(1)]],
                             device const char* vc [[buffer(2)]],
                             device const bfloat* ks [[buffer(3)]],
                             device const bfloat* vs [[buffer(4)]],
                             device float* out [[buffer(5)]],
                             device const uint* positions [[buffer(6)]],
                             constant uint& max_seq [[buffer(7)]],
                             constant uint& kvh [[buffer(8)]],
                             constant uint& window [[buffer(9)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const uint slot = tg / kvh, h = tg % kvh;
    const uint warp = tid / 32u, lane = tid % 32u;
    const uint qh = h * 4u + warp;
    const uint d0 = lane * 4u;
    device const bfloat* qs = q + (ulong)slot * 2048ul;
    device const char* ks8 = kc + ((ulong)slot * kvh + h) * (ulong)max_seq * 128ul;
    device const char* vs8 = vc + ((ulong)slot * kvh + h) * 128ul * (ulong)max_seq;
    device const bfloat* ksc = ks + ((ulong)slot * kvh + h) * (ulong)max_seq;
    device const bfloat* vsc = vs + ((ulong)slot * kvh + h) * (ulong)max_seq;
    const uint pos = positions[slot];
    const uint keys = pos + 1u;
    const uint start = (window && keys > window) ? keys - window : 0u;
    float q0 = float(qs[qh * 128u + d0 + 0u]);
    float q1 = float(qs[qh * 128u + d0 + 1u]);
    float q2 = float(qs[qh * 128u + d0 + 2u]);
    float q3 = float(qs[qh * 128u + d0 + 3u]);
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint t = start; t < keys; ++t) {
        const ulong kb = (ulong)t * 128ul + d0;
        float s = q0 * float(ks8[kb + 0u]) + q1 * float(ks8[kb + 1u]) +
                  q2 * float(ks8[kb + 2u]) + q3 * float(ks8[kb + 3u]);
        s = simd_sum(s) * float(ksc[t]) * 0.08838834764831845f;
        const float nm = max(m, s);
        const float alpha = exp(m - nm);
        const float p = exp(s - nm);
        l = l * alpha + p;
        const float vscale = float(vsc[t]);
        const ulong vb = (ulong)d0 * (ulong)max_seq + t;
        a0 = a0 * alpha + p * float(vs8[vb + 0ul * max_seq]) * vscale;
        a1 = a1 * alpha + p * float(vs8[vb + 1ul * max_seq]) * vscale;
        a2 = a2 * alpha + p * float(vs8[vb + 2ul * max_seq]) * vscale;
        a3 = a3 * alpha + p * float(vs8[vb + 3ul * max_seq]) * vscale;
        m = nm;
    }
    const float inv = 1.0f / l;
    device float* o = out + (ulong)slot * 2048ul + qh * 128u + d0;
    o[0u] = a0 * inv;
    o[1u] = a1 * inv;
    o[2u] = a2 * inv;
    o[3u] = a3 * inv;
}
