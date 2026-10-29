#pragma once
#include <Metal/Metal.hpp>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace miori {

struct Tensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
    const uint8_t* ptr = nullptr;

    int64_t rows() const { return shape.empty() ? 0 : shape[0]; }
    int64_t cols() const { return shape.size() < 2 ? 0 : shape[1]; }
    size_t bytes() const { return (size_t)(end - begin); }
};

class Weights {
public:
    ~Weights();
    bool load(MTL::Device* dev, const std::string& path, std::string& err);
    const Tensor& get(const std::string& name) const;
    const Tensor* find(const std::string& name) const;
    bool has(const std::string& name) const;
    MTL::Buffer* buffer() const { return buf_; }
    const uint8_t* data() const { return data_; }
    size_t data_size() const { return data_size_; }
    size_t tensor_count() const { return tensors_.size(); }
    const std::map<std::string, Tensor>& tensors() const { return tensors_; }

private:
    MTL::Buffer* buf_ = nullptr;
    const uint8_t* data_ = nullptr;
    size_t data_size_ = 0;
    std::map<std::string, Tensor> tensors_;
};

}  // namespace miori
