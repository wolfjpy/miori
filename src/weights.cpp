#include "weights.h"
#include "json.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace miori {

Weights::~Weights() {
    if (buf_) buf_->release();
}

bool Weights::load(MTL::Device* dev, const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    f.seekg(0, std::ios::end);
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    uint64_t header_len = 0;
    f.read((char*)&header_len, 8);
    if (!f || header_len == 0 || header_len > 16 * 1024 * 1024) {
        err = "bad safetensors header length";
        return false;
    }
    std::string header(header_len, '\0');
    f.read(header.data(), (std::streamsize)header_len);
    if (!f) { err = "truncated header"; return false; }
    size_t data_start = 8 + (size_t)header_len;
    if (data_start > file_size) { err = "bad data start"; return false; }
    data_size_ = file_size - data_start;

    Json root;
    try {
        root = JsonParser::parse(header);
    } catch (const std::exception& e) {
        err = std::string("header json: ") + e.what();
        return false;
    }

    buf_ = dev->newBuffer(data_size_, MTL::ResourceStorageModeShared);
    if (!buf_) { err = "MTLBuffer allocation failed for weights"; return false; }
    data_ = (const uint8_t*)buf_->contents();
    f.seekg((std::streamoff)data_start, std::ios::beg);
    f.read((char*)data_, (std::streamsize)data_size_);
    if (!f) { err = "truncated data"; return false; }

    for (const auto& kv : root.obj) {
        if (kv.first == "__metadata__") continue;
        const Json& v = kv.second;
        Tensor t;
        t.name = kv.first;
        t.dtype = v["dtype"].as_string();
        for (const auto& d : v["shape"].arr) t.shape.push_back(d.as_int());
        t.begin = (uint64_t)v["data_offsets"][0].as_int();
        t.end = (uint64_t)v["data_offsets"][1].as_int();
        if (t.end > data_size_ || t.begin > t.end) {
            err = "tensor out of range: " + t.name;
            return false;
        }
        t.ptr = data_ + t.begin;
        tensors_.emplace(t.name, std::move(t));
    }
    printf("weights: %zu tensors, %.2f GB\n", tensors_.size(), data_size_ / 1e9);
    return true;
}

const Tensor& Weights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second;
}

const Tensor* Weights::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool Weights::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

}  // namespace miori
