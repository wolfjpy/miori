#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "metal.h"
#include <cstdio>
#include <stdexcept>

namespace miori {

bool Metal::init(const std::string& metallib_path, std::string& err) {
    dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { err = "no Metal device"; return false; }
    queue = dev->newCommandQueue();
    if (!queue) { err = "no command queue"; return false; }
    NS::Error* e = nullptr;
    NS::String* p = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(p);
    library = dev->newLibrary(url, &e);
    if (!library) {
        err = e ? e->localizedDescription()->utf8String() : "failed to load metallib";
        return false;
    }
    return true;
}

MTL::ComputePipelineState* Metal::pso(const std::string& name) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    NS::String* n = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    MTL::Function* fn = library->newFunction(n);
    if (!fn) throw std::runtime_error("missing shader: " + name);
    NS::Error* e = nullptr;
    MTL::ComputePipelineState* p = dev->newComputePipelineState(fn, &e);
    if (!p) {
        throw std::runtime_error("pipeline " + name + ": " +
                                 (e ? e->localizedDescription()->utf8String() : "?"));
    }
    pipelines[name] = p;
    return p;
}

MTL::Buffer* Metal::new_buffer(size_t bytes, bool shared) {
    MTL::ResourceOptions opt = shared ? MTL::ResourceStorageModeShared
                                     : MTL::ResourceStorageModePrivate;
    MTL::Buffer* b = dev->newBuffer(bytes, opt);
    if (!b) throw std::runtime_error("MTLBuffer allocation failed");
    return b;
}

MTL::CommandBuffer* Metal::begin() {
    MTL::CommandBuffer* cb = queue->commandBuffer();
    if (!cb) throw std::runtime_error("command buffer");
    return cb;
}

void Metal::submit(MTL::CommandBuffer* cb, bool wait) {
    cb->commit();
    if (wait) cb->waitUntilCompleted();
}

void Metal::print_info() const {
    printf("device: %s\n", dev->name()->utf8String());
    printf("recommended working set: %.2f GB\n", dev->recommendedMaxWorkingSetSize() / 1e9);
    printf("max buffer: %.2f GB\n", dev->maxBufferLength() / 1e9);
    printf("threadgroup memory: %lu\n", (unsigned long)dev->maxThreadgroupMemoryLength());
}

}  // namespace miori
