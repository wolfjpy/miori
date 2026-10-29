#pragma once
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace miori {

struct Metal {
    MTL::Device* dev = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* library = nullptr;
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines;

    bool init(const std::string& metallib_path, std::string& err);
    MTL::ComputePipelineState* pso(const std::string& name);
    MTL::Buffer* new_buffer(size_t bytes, bool shared = true);
    MTL::CommandBuffer* begin();
    void submit(MTL::CommandBuffer* cb, bool wait = true);
    void print_info() const;
};

}  // namespace miori
