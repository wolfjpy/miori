// LFM3-VL vision tower + multimodal projector.
//
// Weights are the bf16 tensors already present in models/model.safetensors
// (vision_tower.* and multi_modal_projector.*). No dequantisation is involved:
// every GEMM reads bf16 weights through the naive bf16 Metal GEMM.
#pragma once
#include "metal.h"
#include "weights.h"
#include <cstdint>
#include <string>
#include <vector>

namespace miori {

struct VisionConfig {
    int hidden = 1024;
    int inter = 2560;
    int layers = 14;
    int heads = 16;
    int kv_heads = 8;
    int head_dim = 64;
    int patch_dim = 1536;      // 3 channels * temporal 2 * 16 * 16
    int patch_size = 16;
    int merge = 2;             // spatial_merge_size / downsample_factor
    int temporal = 2;
    int proj_hidden = 4096;
    float norm_eps = 1e-5f;
    float proj_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

struct ImageInfo {
    int src_w = 0, src_h = 0;          // decoded size
    int resized_w = 0, resized_h = 0;  // after smart_resize (multiples of 32)
    int grid_w = 0, grid_h = 0;        // patch grid (multiples of 2)
    int n_patches = 0;                 // grid_w * grid_h
    int n_tokens = 0;                  // n_patches / 4 == (H/32)*(W/32)
};

// ImageIO/CoreGraphics decode -> smart_resize (factor 32) -> antialiased
// bicubic -> (x/255 - 0.5)/0.5 -> patchify in merge-block order with the
// temporal pair duplicated. `patches` is bf16 [n_patches][1536] on return.
bool preprocess_image(const std::string& path, int min_pixels, int max_pixels,
                      std::vector<uint16_t>& patches, ImageInfo& info, std::string& err,
                      std::vector<uint8_t>* resized_rgb = nullptr);

class Vision {
public:
    Vision(Metal& m, Weights& w) : m_(m), w_(w) {}
    ~Vision();

    bool init(std::string& err);

    // Runs the tower + projector. `features` is bf16 [n_tokens][2048].
    bool encode(const std::vector<uint16_t>& patches, const ImageInfo& info,
                std::vector<uint16_t>& features, std::string& err);

    // Independent scalar C++/fp32 implementation of the same computation,
    // written from the reference module semantics. Used by vision-test to
    // check the GPU layout (conv permutation, axial RoPE, merge order).
    bool cpu_reference(const std::vector<uint16_t>& patches, const ImageInfo& info,
                       std::vector<float>& features, std::string& err);

    static const char* layer_type(int i);   // "conv" or "attention"
    static int image_tokens(int resized_h, int resized_w) {
        return (resized_h / 32) * (resized_w / 32);
    }

private:
    Metal& m_;
    Weights& w_;
    VisionConfig cfg_;
    int cap_patches_ = 0;

    MTL::Buffer* patches_ = nullptr;   // bf16 [N][1536]
    MTL::Buffer* x_ = nullptr;         // f32  [N][1024] residual stream
    MTL::Buffer* h_ = nullptr;         // bf16 [N][1024]
    MTL::Buffer* qf_ = nullptr;        // f32  [N][1024]
    MTL::Buffer* kf_ = nullptr;        // f32  [N][512]
    MTL::Buffer* vf_ = nullptr;        // f32  [N][512]
    MTL::Buffer* qn_ = nullptr;        // bf16 [N][1024]
    MTL::Buffer* kn_ = nullptr;        // bf16 [N][512]
    MTL::Buffer* vt_ = nullptr;        // bf16 [512][N]
    MTL::Buffer* attn_ = nullptr;      // f32  [N][1024]
    MTL::Buffer* attn_b_ = nullptr;    // bf16 [N][1024]
    MTL::Buffer* proj_ = nullptr;      // f32  [N][1024]
    MTL::Buffer* bcx_ = nullptr;       // f32  [N][3072]
    MTL::Buffer* gated_ = nullptr;     // f32  [N][1024]
    MTL::Buffer* conv_ = nullptr;      // f32  [N][1024]
    MTL::Buffer* conv_b_ = nullptr;    // bf16 [N][1024]
    MTL::Buffer* fg_ = nullptr;        // f32  [N][2560]
    MTL::Buffer* fu_ = nullptr;        // f32  [N][2560]
    MTL::Buffer* fact_ = nullptr;      // bf16 [N][2560]
    MTL::Buffer* down_ = nullptr;      // f32  [N][1024]
    MTL::Buffer* scores_ = nullptr;    // f32  [128][N]
    MTL::Buffer* probs_ = nullptr;     // bf16 [128][N]
    MTL::Buffer* ln_ = nullptr;        // bf16 [N][1024]
    MTL::Buffer* lin1_ = nullptr;      // f32  [N/4][4096]
    MTL::Buffer* gelu_ = nullptr;      // bf16 [N/4][4096]
    MTL::Buffer* lin2_ = nullptr;      // f32  [N/4][2048]
    MTL::Buffer* feat_ = nullptr;      // bf16 [N/4][2048]

    void ensure(int n_patches);
};

}  // namespace miori
