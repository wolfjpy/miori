// Tensor-core GEMM (MPP matmul2d) with 4-bit weights dequantised into a
// tile-major bf16 layout [N/64][K][64].
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void dequant_tile(device const uint4* q [[buffer(0)]],
                         device const bfloat* s [[buffer(1)]],
                         device const bfloat* b [[buffer(2)]],
                         device bfloat* out [[buffer(3)]],
                         constant uint& N [[buffer(4)]],
                         constant uint& K [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const uint kb = tg.x, nt = tg.y;
    const uint r = tid >> 2;
    const uint seg = tid & 3u;
    const uint row = nt * 64u + r;
    device const uint4* qrow = q + (ulong)row * (K / 32u);
    device const bfloat* srow = s + (ulong)row * (K / 32u);
    device const bfloat* brow = b + (ulong)row * (K / 32u);
    device bfloat* o = out + (ulong)nt * (ulong)K * 64ul + (ulong)kb * 128ul * 64ul;
    for (uint e = seg * 32u; e < seg * 32u + 32u; ++e) {
        const uint k = kb * 128u + e;
        const uint g = k / 32u;
        const uint4 w = qrow[g];
        const uint ww[4] = {w.x, w.y, w.z, w.w};
        const uint word = ww[(k % 32u) / 8u];
        const float qv = float((word >> (4u * (k % 8u))) & 0xFu);
        o[(ulong)e * 64ul + r] = bfloat(qv * float(srow[g]) + float(brow[g]));
    }
}

kernel void gemm_mpp(device bfloat* A [[buffer(0)]],
                     device bfloat* B [[buffer(1)]],
                     device float* C [[buffer(2)]],
                     constant uint& M [[buffer(3)]],
                     constant uint& N [[buffer(4)]],
                     constant uint& K [[buffer(5)]],
                     constant uint& lda [[buffer(6)]],
                     constant uint& ldc [[buffer(7)]],
                     uint2 tgid [[threadgroup_position_in_grid]]) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(int(K), int(M)));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(int(N), int(M)));
    constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, int(tgid.y) * 64);
    auto mC = tC.slice(int(tgid.x) * 64, int(tgid.y) * 64);
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tB(
        B + (ulong)tgid.x * (ulong)K * 64ul, dextents<int32_t, 2>(64, int(K)));
    op.run(mA, tB, mC);
}
