#pragma once
#include "weights.h"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace miori {

inline float bf16_to_float(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

inline uint16_t float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    // round to nearest even
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

inline float dequant4_elem(const Tensor& q, const Tensor& s, const Tensor& b,
                           int64_t n, int64_t k, int64_t K) {
    const uint32_t* qrow = (const uint32_t*)(q.ptr + (uint64_t)n * (uint64_t)q.cols() * 4);
    const uint32_t word = qrow[k / 8];
    const uint32_t qv = (word >> (4 * (k % 8))) & 0xF;
    const int64_t g = k / 32;
    const float sc = bf16_to_float(((const uint16_t*)s.ptr)[(uint64_t)n * (K / 32) + g]);
    const float bi = bf16_to_float(((const uint16_t*)b.ptr)[(uint64_t)n * (K / 32) + g]);
    return (float)qv * sc + bi;
}

inline float dequant8_elem(const Tensor& q, const Tensor& s, const Tensor& b,
                           int64_t n, int64_t k, int64_t K) {
    const uint32_t* qrow = (const uint32_t*)(q.ptr + (uint64_t)n * (uint64_t)q.cols() * 4);
    const uint32_t word = qrow[k / 4];
    const uint32_t qv = (word >> (8 * (k % 4))) & 0xFF;
    const int64_t g = k / 32;
    const float sc = bf16_to_float(((const uint16_t*)s.ptr)[(uint64_t)n * (K / 32) + g]);
    const float bi = bf16_to_float(((const uint16_t*)b.ptr)[(uint64_t)n * (K / 32) + g]);
    return (float)qv * sc + bi;
}

}  // namespace miori
