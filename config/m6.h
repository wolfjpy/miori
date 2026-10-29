// Miori — compile-time constants for Apple M6 / LFM3-VL-2B-MLX-4bit.
// Shared by C++ host and Metal shaders.
#pragma once
#ifdef __METAL_VERSION__
  #include <metal_stdlib>
  #define MIORI_CONST constant constexpr
  using miori_u32 = uint32_t;
#else
  #include <cstdint>
  #define MIORI_CONST inline constexpr
  using miori_u32 = uint32_t;
#endif

namespace miori {

// ---- hardware (Apple M6, 12 GPU cores, 16 GB) ----
MIORI_CONST miori_u32 kGpuCores = 12;
MIORI_CONST miori_u32 kThreadgroupMem = 32768;
MIORI_CONST miori_u32 kSimdWidth = 32;

// ---- text model ----
MIORI_CONST miori_u32 kLayers = 24;
MIORI_CONST miori_u32 kHidden = 2048;
MIORI_CONST miori_u32 kFfn = 7680;
MIORI_CONST miori_u32 kVocab = 128000;
MIORI_CONST miori_u32 kHeads = 16;
MIORI_CONST miori_u32 kKvHeads = 4;
MIORI_CONST miori_u32 kHeadDim = 128;
MIORI_CONST miori_u32 kQDim = kHeads * kHeadDim;       // 2048
MIORI_CONST miori_u32 kKvDim = kKvHeads * kHeadDim;    // 512
MIORI_CONST miori_u32 kConvK = 7;
MIORI_CONST miori_u32 kSliding = 256;
MIORI_CONST float kRmsEps = 1e-5f;
MIORI_CONST float kRopeThetaFull = 1000000.0f;
MIORI_CONST float kRopeThetaSliding = 10000.0f;
MIORI_CONST float kAttnScale = 0.08838834764831845f;   // 1/sqrt(128)

// 0 = conv, 1 = full attention, 2 = sliding attention
MIORI_CONST miori_u32 kLayerType[kLayers] = {
    0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0,
    2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0,
};

MIORI_CONST miori_u32 kNumFull = 6;
MIORI_CONST miori_u32 kNumSliding = 5;

// ---- quantisation ----
MIORI_CONST miori_u32 kGroup = 32;

// ---- runtime limits ----
MIORI_CONST miori_u32 kMaxSeq = 4096;       // per-request KV capacity for v0.1
MIORI_CONST miori_u32 kMaxBatch = 64;
MIORI_CONST miori_u32 kPrefillChunk = 128;

// ---- tokens ----
MIORI_CONST int kBosId = 124894;
MIORI_CONST int kEosId = 124900;   // <|im_end|>
MIORI_CONST int kPadId = 124893;
MIORI_CONST int kImStartId = 124899;
MIORI_CONST int kImEndId = 124900;
MIORI_CONST int kThinkId = 124901;
MIORI_CONST int kEndThinkId = 124902;
MIORI_CONST int kImageId = 125030;

}  // namespace miori
