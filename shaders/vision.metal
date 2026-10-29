// LFM3-VL vision tower kernels.
//
// The vision tower is a stack of gated depthwise 3x3 convolutions and
// bidirectional grouped-query attention layers over the patch grid of one
// image. All weights are bf16, all accumulation is fp32. Activations are kept
// fp32 in the residual stream and bf16 wherever they feed a GEMM, which mirrors
// the bf16 activations of the reference implementation while avoiding a
// needless precision loss in the residual path.
//
// The existing 1024-wide RMSNorm (f32 in -> bf16 out) and the naive bf16 GEMM
// from ops.metal are reused; this file only adds what the vision tower needs:
// LayerNorm with bias, exact GELU, bidirectional softmax, the axial 2D RoPE,
// the merge-block <-> raster permutation for the spatial convolution, and the
// V transpose used by the attention matmul.

#include <metal_stdlib>
#include "m6.h"

using namespace metal;

// ---------------------------------------------------------------- erf
// Metal has no erf(); this is Abramowitz & Stegun 7.1.26 (|err| <= 1.5e-7),
// far below bf16 resolution, so GELU is "exact" for this model.
static inline float miori_erf(float x) {
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                             - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return x < 0.0f ? -y : y;
}

static inline float miori_gelu(float x) {
    return 0.5f * x * (1.0f + miori_erf(x * 0.7071067811865476f));
}

// ---------------------------------------------------------------- LayerNorm
// bf16 in -> bf16 out, weight + bias, one threadgroup (256 threads) per row.
kernel void vision_layernorm(device const bfloat* x [[buffer(0)]],
                             device const bfloat* w [[buffer(1)]],
                             device const bfloat* b [[buffer(2)]],
                             device bfloat* y [[buffer(3)]],
                             constant uint& n [[buffer(4)]],
                             constant float& eps [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[16];
    const ulong ro = (ulong)tg * n;
    float sum = 0.0f, sq = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        const float v = float(x[ro + i]);
        sum += v;
        sq += v * v;
    }
    sum = simd_sum(sum);
    sq = simd_sum(sq);
    if (lane == 0u) { part[warp] = sum; part[warp + 8u] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float s = 0.0f, q = 0.0f;
        for (uint i = 0; i < 8u; ++i) { s += part[i]; q += part[i + 8u]; }
        const float mean = s / float(n);
        const float var = max(q / float(n) - mean * mean, 0.0f);
        part[0] = mean;
        part[1] = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = part[0], inv = part[1];
    for (uint i = tid; i < n; i += 256u)
        y[ro + i] = bfloat((float(x[ro + i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// ---------------------------------------------------------------- bias / GELU
kernel void vision_bias(device const float* x [[buffer(0)]],
                        device const bfloat* bias [[buffer(1)]],
                        device bfloat* y [[buffer(2)]],
                        constant uint& total [[buffer(3)]],
                        constant uint& cols [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= total) return;
    y[i] = bfloat(x[i] + float(bias[i % cols]));
}

kernel void vision_bias_gelu(device const float* x [[buffer(0)]],
                             device const bfloat* bias [[buffer(1)]],
                             device bfloat* y [[buffer(2)]],
                             constant uint& total [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= total) return;
    y[i] = bfloat(miori_gelu(x[i] + float(bias[i % cols])));
}

// ---------------------------------------------------------------- bidirectional softmax
// probs[row, s] = softmax(scores[row, s] * scale) over the whole row; one
// threadgroup (256 threads) per row. Attention inside one image is not causal.
kernel void vision_softmax(device float* scores [[buffer(0)]],
                           device bfloat* probs [[buffer(1)]],
                           constant uint& seq [[buffer(2)]],
                           constant float& scale [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const ulong ro = (ulong)row * seq;
    float mx = -INFINITY;
    for (uint s = tid; s < seq; s += 256u) mx = max(mx, scores[ro + s] * scale);
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
        const float e = exp(scores[ro + s] * scale - mx);
        scores[ro + s] = e;
        sum += e;
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
        probs[ro + s] = bfloat(scores[ro + s] * inv);
}

// ---------------------------------------------------------------- axial 2D RoPE
// One thread per (patch, head, frequency pair). Patches are stored in
// merge-block order; the (h, w) coordinate of a patch is
//   h = block_row*2 + (p%4)/2,  w = block_col*2 + p%2.
// The reference rotates the pair (d, d+32) of each 64-wide head by
//   angle = pos(axis) * theta^(-i/16),  i = d % 16,
//   axis = h for d % 32 < 16, w otherwise.
kernel void vision_rope(device bfloat* x [[buffer(0)]],
                        constant uint& n [[buffer(1)]],
                        constant uint& heads [[buffer(2)]],
                        constant uint& dim [[buffer(3)]],
                        constant uint& gw [[buffer(4)]],
                        constant float& theta [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    const uint pairs = dim >> 1;
    if (i >= n * heads * pairs) return;
    const uint p = i / (heads * pairs);
    const uint r = i % (heads * pairs);
    const uint head = r / pairs;
    const uint d = r % pairs;                 // 0 .. 31
    const uint bpr = gw >> 1;
    const uint bh = (p >> 2) / bpr;
    const uint bw = (p >> 2) % bpr;
    const float h = float(bh * 2u + ((p >> 1) & 1u));
    const float w = float(bw * 2u + (p & 1u));
    const uint fi = d % 16u;
    const float inv = exp2(-float(fi) * (log2(theta) / 16.0f));
    const float ang = ((d % 32u) < 16u ? h : w) * inv;
    const float c = cos(ang), s = sin(ang);
    const uint base = p * heads * dim + head * dim;
    const float x0 = float(x[base + d]);
    const float x1 = float(x[base + d + 32u]);
    x[base + d] = bfloat(x0 * c - x1 * s);
    x[base + d + 32u] = bfloat(x1 * c + x0 * s);
}

// ---------------------------------------------------------------- V transpose
// v is [n][d] bf16 -> vt is [d][n] bf16; the attention matmul needs V in
// [output_dim][seq] layout because gemm_naive computes A[M,K] * B[N,K]^T.
kernel void vision_vt(device const float* v [[buffer(0)]],
                      device bfloat* vt [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      constant uint& d [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n * d) return;
    const uint s = i / d, c = i % d;
    vt[(ulong)c * n + s] = bfloat(v[i]);
}

// ---------------------------------------------------------------- gated 3x3 depthwise conv
// out[p, c] = C[p, c] * sum_{kh,kw} gated[raster(h+kh-1, w+kw-1), c] * W[c, kh, kw]
// where p is the patch index in merge-block order, (h, w) its raster
// coordinate and the conv is zero-padded by one patch. gated = B * x is already
// in merge-block order (conv_gate); the C half of in_proj is applied here.
kernel void vision_conv3x3(device const float* gated [[buffer(0)]],
                           device const float* bcx [[buffer(1)]],
                           device const bfloat* wt [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant uint& n [[buffer(4)]],
                           constant uint& gw [[buffer(5)]],
                           constant uint& gh [[buffer(6)]],
                           constant uint& H [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n * H) return;
    const uint p = i / H, c = i % H;
    const uint bpr = gw >> 1;
    const uint bh = (p >> 2) / bpr, bw = (p >> 2) % bpr;
    const int hh = int(bh * 2u + ((p >> 1) & 1u));
    const int ww = int(bw * 2u + (p & 1u));
    float acc = 0.0f;
    for (int kh = -1; kh <= 1; ++kh) {
        const int nh = hh + kh;
        if (nh < 0 || nh >= int(gh)) continue;
        for (int kw = -1; kw <= 1; ++kw) {
            const int nw = ww + kw;
            if (nw < 0 || nw >= int(gw)) continue;
            const uint np = (((uint(nh) >> 1) * bpr + (uint(nw) >> 1)) << 2)
                            | ((uint(nh) & 1u) << 1) | (uint(nw) & 1u);
            acc += gated[(ulong)np * H + c] * float(wt[c * 9u + uint(kh + 1) * 3u + uint(kw + 1)]);
        }
    }
    out[i] = bcx[(ulong)p * 3u * H + H + c] * acc;
}
