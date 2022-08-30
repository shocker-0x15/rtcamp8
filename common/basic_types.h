﻿#pragma once

#include "common_shared.h"

#define INCLUDE_CUDA_RUNTIME_API 1
#if INCLUDE_CUDA_RUNTIME_API
#include <cuda_runtime_api.h>
#endif

#define vec2print(v) (v).x, (v).y
#define vec3print(v) (v).x, (v).y, (v).z
#define vec4print(v) (v).x, (v).y, (v).z, (v).w
#define rgbprint(v) (v).r, (v).g, (v).b

#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__)
// ----------------------------------------------------------------
// JP: CUDAビルトインに対応する型・関数をホスト側で定義しておく。
// EN: Define types and functions on the host corresponding to CUDA built-ins.

#if !INCLUDE_CUDA_RUNTIME_API
struct alignas(8) int2 {
    int32_t x, y;
    constexpr int2(int32_t v = 0) : x(v), y(v) {}
    constexpr int2(int32_t xx, int32_t yy) : x(xx), y(yy) {}
};
struct int3 {
    int32_t x, y, z;
    constexpr int3(int32_t v = 0) : x(v), y(v), z(v) {}
    constexpr int3(int32_t xx, int32_t yy, int32_t zz) : x(xx), y(yy), z(zz) {}
};
struct alignas(16) int4 {
    int32_t x, y, z, w;
    constexpr int4(int32_t v = 0) : x(v), y(v), z(v), w(v) {}
    constexpr int4(int32_t xx, int32_t yy, int32_t zz, int32_t ww) : x(xx), y(yy), z(zz), w(ww) {}
};
struct alignas(8) uint2 {
    uint32_t x, y;
    constexpr uint2(uint32_t v = 0) : x(v), y(v) {}
    constexpr uint2(uint32_t xx, uint32_t yy) : x(xx), y(yy) {}
};
struct uint3 {
    uint32_t x, y, z;
    constexpr uint3(uint32_t v = 0) : x(v), y(v), z(v) {}
    constexpr uint3(uint32_t xx, uint32_t yy, uint32_t zz) : x(xx), y(yy), z(zz) {}
};
struct uint4 {
    uint32_t x, y, z, w;
    constexpr uint4(uint32_t v = 0) : x(v), y(v), z(v), w(v) {}
    constexpr uint4(uint32_t xx, uint32_t yy, uint32_t zz, uint32_t ww) : x(xx), y(yy), z(zz), w(ww) {}
};
struct alignas(8) float2 {
    float x, y;
    constexpr float2(float v = 0) : x(v), y(v) {}
    constexpr float2(float xx, float yy) : x(xx), y(yy) {}
};
struct float3 {
    float x, y, z;
    constexpr float3(float v = 0) : x(v), y(v), z(v) {}
    constexpr float3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    constexpr float3(const uint3 &v) :
        x(static_cast<float>(v.x)), y(static_cast<float>(v.y)), z(static_cast<float>(v.z)) {}
};
struct alignas(16) float4 {
    float x, y, z, w;
    constexpr float4(float v = 0) : x(v), y(v), z(v), w(v) {}
    constexpr float4(float xx, float yy, float zz, float ww) : x(xx), y(yy), z(zz), w(ww) {}
    constexpr float4(const float3 &xyz, float ww) : x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}
};
#endif

inline constexpr int2 make_int2(int32_t x, int32_t y) {
    return int2(x, y);
}
inline constexpr int3 make_int3(int32_t x, int32_t y, int32_t z) {
    return int3(x, y, z);
}
inline constexpr int4 make_int4(int32_t x, int32_t y, int32_t z, int32_t w) {
    return int4(x, y, z, w);
}
inline constexpr uint2 make_uint2(uint32_t x, uint32_t y) {
    return uint2(x, y);
}
inline constexpr uint3 make_uint3(uint32_t x, uint32_t y, uint32_t z) {
    return uint3(x, y, z);
}
inline constexpr uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return uint4(x, y, z, w);
}
inline float2 make_float2(float x, float y) {
    return float2(x, y);
}
inline constexpr float3 make_float3(float x, float y, float z) {
    return float3(x, y, z);
}
inline constexpr float4 make_float4(float x, float y, float z, float w) {
    return float4(x, y, z, w);
}

// END: Define types and functions on the host corresponding to CUDA built-ins.
// ----------------------------------------------------------------
#endif

CUDA_COMMON_FUNCTION CUDA_INLINE float3 getXYZ(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(const float2 &v) {
    return make_int2(static_cast<int32_t>(v.x), static_cast<int32_t>(v.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(const int3 &v) {
    return make_int2(v.x, v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 make_int2(const uint3 &v) {
    return make_int2(static_cast<int32_t>(v.x), static_cast<int32_t>(v.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const int2 &v0, const int2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const int2 &v0, const int2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const int2 &v0, const uint2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const int2 &v0, const uint2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const int2 &v0, const uint2 &v1) {
    return make_uint2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator+(const int2 &v0, const int2 &v1) {
    return make_int2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &v0, const int2 &v1) {
    return make_int2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(uint32_t s, const int2 &v) {
    return make_int2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator*(const int2 &v, uint32_t s) {
    return make_int2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &v0, const int2 &v1) {
    v0.x *= v1.x;
    v0.y *= v1.y;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 &operator*=(int2 &v, uint32_t s) {
    v.x *= s;
    v.y *= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/(const int2 &v0, const int2 &v1) {
    return make_int2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 operator/(const int2 &v, uint32_t s) {
    return make_int2(v.x / s, v.y / s);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const int2 &v0, const uint2 &v1) {
    return make_uint2(v0.x / v1.x, v0.y / v1.y);
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 make_uint2(const float2 &v) {
    return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 make_uint2(const int3 &v) {
    return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 make_uint2(const uint3 &v) {
    return make_uint2(v.x, v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const uint2 &v0, const uint2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const uint2 &v0, const uint2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const uint2 &v0, const int2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const uint2 &v0, const int2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator+(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator+=(uint2 &v, uint32_t s) {
    v.x += s;
    v.y += s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator-(const uint2 &v, uint32_t s) {
    return make_uint2(v.x - s, v.y - s);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator-=(uint2 &v, uint32_t s) {
    v.x -= s;
    v.y -= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(float s, const uint2 &v) {
    return make_uint2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator*(const uint2 &v, float s) {
    return make_uint2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator*=(uint2 &v0, const uint2 &v1) {
    v0.x *= v1.x;
    v0.y *= v1.y;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator*=(uint2 &v, uint32_t s) {
    v.x *= s;
    v.y *= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &v0, const int2 &v1) {
    return make_uint2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator/(const uint2 &v, uint32_t s) {
    return make_uint2(v.x / s, v.y / s);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator/=(uint2 &v, uint32_t s) {
    v.x /= s;
    v.y /= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator%(const uint2 &v0, const uint2 &v1) {
    return make_uint2(v0.x % v1.x, v0.y % v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator<<(const uint2 &v, uint32_t s) {
    return make_uint2(v.x << s, v.y << s);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator<<=(uint2 &v, uint32_t s) {
    v = v << s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 operator>>(const uint2 &v, uint32_t s) {
    return make_uint2(v.x >> s, v.y >> s);
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 &operator>>=(uint2 &v, uint32_t s) {
    v = v >> s;
    return v;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float2 make_float2(float v) {
    return make_float2(v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 make_float2(const int2 &v) {
    return make_float2(v.x, v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 make_float2(const uint2 &v) {
    return make_float2(v.x, v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const float2 &v0, const float2 &v1) {
    return v0.x == v1.x && v0.y == v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const float2 &v0, const float2 &v1) {
    return v0.x != v1.x || v0.y != v1.y;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator-(const float2 &v) {
    return make_float2(-v.x, -v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator+(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator-(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x - v1.x, v0.y - v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(float s, const float2 &v) {
    return make_float2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(const float2 &v, float s) {
    return make_float2(s * v.x, s * v.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 &operator*=(float2 &v, float s) {
    v = v * s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(const int2 &v0, const float2 &v1) {
    return make_float2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator*(const float2 &v0, const int2 &v1) {
    return make_float2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator/(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator/(const float2 &v0, const int2 &v1) {
    return make_float2(v0.x / v1.x, v0.y / v1.y);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 operator/(const float2 &v, float s) {
    float r = 1 / s;
    return r * v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 &operator/=(float2 &v, float s) {
    v = v / s;
    return v;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float3 make_float3(float v) {
    return make_float3(v, v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 make_float3(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const float3 &v0, const float3 &v1) {
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const float3 &v0, const float3 &v1) {
    return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator-(const float3 &v) {
    return make_float3(-v.x, -v.y, -v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator+(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator+=(float3 &v0, const float3 &v1) {
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator-(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator-=(float3 &v0, const float3 &v1) {
    v0.x -= v1.x;
    v0.y -= v1.y;
    v0.z -= v1.z;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator*(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator*(float s, const float3 &v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator*(const float3 &v, float s) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator*=(float3 &v0, const float3 &v1) {
    v0.x *= v1.x;
    v0.y *= v1.y;
    v0.z *= v1.z;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator*=(float3 &v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator/(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 operator/(const float3 &v, float s) {
    float r = 1 / s;
    return r * v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 safeDivide(const float3 &v0, const float3 &v1) {
    return make_float3(
        v1.x != 0.0f ? v0.x / v1.x : 0.0f,
        v1.y != 0.0f ? v0.y / v1.y : 0.0f,
        v1.z != 0.0f ? v0.z / v1.z : 0.0f);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 safeDivide(const float3 &v, float d) {
    return d != 0.0f ? (v / d) : make_float3(0.0f);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 &operator/=(float3 &v, float s) {
    float r = 1 / s;
    return v *= r;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(const float3 &v) {
#if !defined(__CUDA_ARCH__)
    using std::isfinite;
#endif
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(float v) {
    return make_float4(v, v, v, v);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(const float3 &v) {
    return make_float4(v.x, v.y, v.z, 0.0f);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 make_float4(const float3 &v, float w) {
    return make_float4(v.x, v.y, v.z, w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator==(const float4 &v0, const float4 &v1) {
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool operator!=(const float4 &v0, const float4 &v1) {
    return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z || v0.w != v1.w;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator-(const float4 &v) {
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator+(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator+=(float4 &v0, const float4 &v1) {
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    v0.w += v1.w;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator-(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator-=(float4 &v0, const float4 &v1) {
    v0.x -= v1.x;
    v0.y -= v1.y;
    v0.z -= v1.z;
    v0.w -= v1.w;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator*(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator*(float s, const float4 &v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator*(const float4 &v, float s) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator*=(float4 &v0, const float4 &v1) {
    v0.x *= v1.x;
    v0.y *= v1.y;
    v0.z *= v1.z;
    v0.w *= v1.w;
    return v0;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator*=(float4 &v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    v.w *= s;
    return v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator/(const float4 &v0, const float4 &v1) {
    return make_float4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 operator/(const float4 &v, float s) {
    float r = 1 / s;
    return r * v;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 &operator/=(float4 &v, float s) {
    float r = 1 / s;
    return v *= r;
}
CUDA_COMMON_FUNCTION CUDA_INLINE bool allFinite(const float4 &v) {
#if !defined(__CUDA_ARCH__)
    using std::isfinite;
#endif
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w);
}

CUDA_COMMON_FUNCTION CUDA_INLINE int2 min(const int2 &v0, const int2 &v1) {
    return make_int2(rtc8::min(v0.x, v1.x),
                     rtc8::min(v0.y, v1.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE int2 max(const int2 &v0, const int2 &v1) {
    return make_int2(rtc8::max(v0.x, v1.x),
                     rtc8::max(v0.y, v1.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint2 min(const uint2 &v0, const uint2 &v1) {
    return make_uint2(rtc8::min(v0.x, v1.x),
                      rtc8::min(v0.y, v1.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE uint2 max(const uint2 &v0, const uint2 &v1) {
    return make_uint2(rtc8::max(v0.x, v1.x),
                      rtc8::max(v0.y, v1.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE float2 min(const float2 &v0, const float2 &v1) {
    return make_float2(rtc8::min(v0.x, v1.x),
                       rtc8::min(v0.y, v1.y));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float2 max(const float2 &v0, const float2 &v1) {
    return make_float2(rtc8::max(v0.x, v1.x),
                       rtc8::max(v0.y, v1.y));
}

CUDA_COMMON_FUNCTION CUDA_INLINE float3 min(const float3 &v0, const float3 &v1) {
    return make_float3(std::fmin(v0.x, v1.x),
                       std::fmin(v0.y, v1.y),
                       std::fmin(v0.z, v1.z));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 max(const float3 &v0, const float3 &v1) {
    return make_float3(std::fmax(v0.x, v1.x),
                       std::fmax(v0.y, v1.y),
                       std::fmax(v0.z, v1.z));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float dot(const float3 &v0, const float3 &v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 cross(const float3 &v0, const float3 &v1) {
    return make_float3(v0.y * v1.z - v0.z * v1.y,
                       v0.z * v1.x - v0.x * v1.z,
                       v0.x * v1.y - v0.y * v1.x);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float squaredDistance(const float3 &p0, const float3 &p1) {
    float3 d = p1 - p0;
    return dot(d, d);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float length(const float3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
CUDA_COMMON_FUNCTION CUDA_INLINE float sqLength(const float3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
CUDA_COMMON_FUNCTION CUDA_INLINE float3 normalize(const float3 &v) {
    return v / length(v);
}

CUDA_COMMON_FUNCTION CUDA_INLINE float4 min(const float4 &v0, const float4 &v1) {
    return make_float4(std::fmin(v0.x, v1.x),
                       std::fmin(v0.y, v1.y),
                       std::fmin(v0.z, v1.z),
                       std::fmin(v0.w, v1.w));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float4 max(const float4 &v0, const float4 &v1) {
    return make_float4(std::fmax(v0.x, v1.x),
                       std::fmax(v0.y, v1.y),
                       std::fmax(v0.z, v1.z),
                       std::fmax(v0.w, v1.w));
}
CUDA_COMMON_FUNCTION CUDA_INLINE float dot(const float4 &v0, const float4 &v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
}



CUDA_COMMON_FUNCTION CUDA_INLINE int32_t floatToOrderedInt(float fVal) {
#if defined(__CUDA_ARCH__)
    int32_t iVal = __float_as_int(fVal);
#else
    int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
#endif
    return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
}

CUDA_COMMON_FUNCTION CUDA_INLINE float orderedIntToFloat(int32_t iVal) {
    int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
#if defined(__CUDA_ARCH__)
    return __int_as_float(orgVal);
#else
    return *reinterpret_cast<float*>(&orgVal);
#endif
}

#if defined(__CUDA_ARCH__) || defined(__INTELLISENSE__)
#   if __CUDA_ARCH__ < 600
#       define atomicOr_block atomicOr
#       define atomicAnd_block atomicAnd
#       define atomicAdd_block atomicAdd
#       define atomicMin_block atomicMin
#       define atomicMax_block atomicMax
#   endif
#endif



namespace rtc8 {

#if __cplusplus < 202002L

template <typename RealType>
static constexpr RealType pi_v = static_cast<RealType>(3.14159265358979323846);

#else

template <typename RealType>
using pi_v = std::numbers::pi_v<RealType>;

#endif

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE bool isnan(RealType x) {
#if defined(__CUDA_ARCH__)
    return static_cast<bool>(::isnan(x));
#else
    return std::isnan(x);
#endif
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE bool isinf(RealType x) {
#if defined(__CUDA_ARCH__)
    return static_cast<bool>(::isinf(x));
#else
    return std::isinf(x);
#endif
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE bool isfinite(RealType x) {
#if defined(__CUDA_ARCH__)
    return static_cast<bool>(::isfinite(x));
#else
    return std::isfinite(x);
#endif
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE void sincos(RealType angle, RealType* s, RealType* c) {
#if defined(__CUDA_ARCH__)
    ::sincosf(angle, s, c);
#else
    *s = std::sin(angle);
    *c = std::cos(angle);
#endif
}



template <typename RealType>
struct CompensatedSum {
    RealType result;
    RealType comp;

    CUDA_COMMON_FUNCTION constexpr CompensatedSum(const RealType &value = RealType(0)) :
        result(value), comp(static_cast<RealType>(0)) { };

    CUDA_COMMON_FUNCTION constexpr CompensatedSum &operator=(const RealType &value) {
        result = value;
        comp = 0;
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr CompensatedSum &operator+=(const RealType &value) {
        RealType cInput = value - comp;
        RealType sumTemp = result + cInput;
        comp = (sumTemp - result) - cInput;
        result = sumTemp;
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr operator RealType() const { return result; };
};



template <typename RealType>
struct Point3DTemplate;

template <typename RealType, bool isNormal = false>
struct Vector3DTemplate;

template <typename RealType>
using Normal3DTemplate = Vector3DTemplate<RealType, true>;

template <typename RealType>
struct Vector4DTemplate;

template <typename RealType>
struct TexCoord2DTemplate;

template <typename RealType>
struct Matrix3x3Template;

template <typename RealType>
struct Matrix4x4Template;

template <typename RealType>
struct QuaternionTemplate;

template <typename RealType>
struct BoundingBox3DTemplate;

template <typename RealType>
struct RGBTemplate;



template <typename RealType>
struct Point3DTemplate {
    RealType x, y, z;

    CUDA_COMMON_FUNCTION Point3DTemplate() {}
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate(RealType v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate(RealType xx, RealType yy, RealType zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION constexpr explicit Point3DTemplate(const float3 &p) :
        x(p.x), y(p.y), z(p.z) {}

    CUDA_COMMON_FUNCTION constexpr Point3DTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate operator-() const {
        return Point3DTemplate(-x, -y, -z);
    }

    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator+=(const Point3DTemplate &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator+=(const Vector3DTemplate<RealType> &v);
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator+=(const Normal3DTemplate<RealType> &v);
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator-=(const Vector3DTemplate<RealType> &v);
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator-=(const Normal3DTemplate<RealType> &v);
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator*=(RealType s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        return *this *= r;
    }

    CUDA_COMMON_FUNCTION constexpr RealType &operator[](uint32_t dim) {
        Assert(dim <= 2, "\"dim\" is out of range [0, 2].");
        return *(&x + dim);
    }
    CUDA_COMMON_FUNCTION constexpr const RealType &operator[](uint32_t dim) const {
        Assert(dim <= 2, "\"dim\" is out of range [0, 2].");
        return *(&x + dim);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc8::isnan;
        return isnan(x) || isnan(y) || isnan(z);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc8::isinf;
        return isinf(x) || isinf(y) || isinf(z);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc8::isfinite;
        return isfinite(x) && isfinite(y) && isfinite(z);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ float3 toNativeType() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Point3DTemplate Zero() {
        return Point3DTemplate(0, 0, 0);
    }
};



template <typename RealType, bool isNormal>
struct Vector3DTemplate {
    RealType x, y, z;

    CUDA_COMMON_FUNCTION Vector3DTemplate() {}
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate(RealType v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate(RealType xx, RealType yy, RealType zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION constexpr explicit Vector3DTemplate(const Point3DTemplate<RealType> &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION constexpr explicit Vector3DTemplate(const float3 &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate(const Vector3DTemplate<RealType, !isNormal> &v) :
        x(v.x), y(v.y), z(v.z) {}
    CUDA_COMMON_FUNCTION constexpr explicit operator Point3DTemplate<RealType>() const {
        return Point3DTemplate<RealType>(x, y, z);
    }

    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate operator-() const {
        return Vector3DTemplate(-x, -y, -z);
    }

    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate &operator+=(const Vector3DTemplate &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate &operator-=(const Vector3DTemplate &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate &operator*=(const Vector3DTemplate &v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate &operator*=(RealType s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate &operator/=(const Vector3DTemplate &v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        return *this *= r;
    }

    CUDA_COMMON_FUNCTION constexpr RealType &operator[](uint32_t dim) {
        Assert(dim <= 2, "\"dim\" is out of range [0, 2].");
        return *(&x + dim);
    }
    CUDA_COMMON_FUNCTION constexpr const RealType &operator[](uint32_t dim) const {
        Assert(dim <= 2, "\"dim\" is out of range [0, 2].");
        return *(&x + dim);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc8::isnan;
        return isnan(x) || isnan(y) || isnan(z);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc8::isinf;
        return isinf(x) || isinf(y) || isinf(z);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc8::isfinite;
        return isfinite(x) && isfinite(y) && isfinite(z);
    }

    CUDA_COMMON_FUNCTION constexpr RealType squaredLength() const {
        return pow2(x) + pow2(y) + pow2(z);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ RealType length() const {
        return std::sqrt(squaredLength());
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ Vector3DTemplate &normalize() {
        *this /= length();
        return *this;
    }

    // References
    // Building an Orthonormal Basis, Revisited
    CUDA_COMMON_FUNCTION constexpr void makeCoordinateSystem(
        Vector3DTemplate<RealType, false>* vx, Vector3DTemplate<RealType, false>* vy) const {
        RealType sign = z >= 0 ? 1 : -1;
        RealType a = -1 / (sign + z);
        RealType b = x * y * a;
        *vx = Vector3DTemplate<RealType, false>(1 + sign * x * x * a, sign * b, -sign * x);
        *vy = Vector3DTemplate<RealType, false>(b, sign + y * y * a, -y);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ void toPolarZUp(RealType* theta, RealType* phi) const {
        *theta = std::acos(clamp(z, static_cast<RealType>(-1), static_cast<RealType>(1)));
        *phi = std::fmod(static_cast<RealType>(std::atan2(y, x) + 2 * pi_v<RealType>),
                         static_cast<RealType>(2 * pi_v<RealType>));
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ void toPolarYUp(RealType* theta, RealType* phi) const {
        *theta = std::acos(clamp(y, static_cast<RealType>(-1), static_cast<RealType>(1)));
        *phi = std::fmod(static_cast<RealType>(std::atan2(-x, z) + 2 * pi_v<RealType>),
                         static_cast<RealType>(2 * pi_v<RealType>));
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ float3 toNativeType() const {
        return make_float3(x, y, z);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector3DTemplate Zero() {
        return Vector3DTemplate(0, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector3DTemplate Ex() {
        return Vector3DTemplate(1, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector3DTemplate Ey() {
        return Vector3DTemplate(0, 1, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector3DTemplate Ez() {
        return Vector3DTemplate(0, 0, 1);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static /*constexpr*/ Vector3DTemplate fromPolarZUp(
        RealType phi, RealType theta) {
        RealType sinPhi, cosPhi;
        RealType sinTheta, cosTheta;
        rtc8::sincos(phi, &sinPhi, &cosPhi);
        rtc8::sincos(theta, &sinTheta, &cosTheta);
        return Vector3DTemplate(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static /*constexpr*/ Vector3DTemplate fromPolarYUp(
        RealType phi, RealType theta) {
        RealType sinPhi, cosPhi;
        RealType sinTheta, cosTheta;
        rtc8::sincos(phi, &sinPhi, &cosPhi);
        rtc8::sincos(theta, &sinTheta, &cosTheta);
        return Vector3DTemplate(-sinPhi * sinTheta, cosTheta, cosPhi * sinTheta);
    }
};



template <typename RealType>
struct Vector4DTemplate {
    RealType x, y, z, w;

    CUDA_COMMON_FUNCTION Vector4DTemplate() {}
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate(RealType v) :
        x(v), y(v), z(v), w(v) {}
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate(RealType xx, RealType yy, RealType zz, RealType ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate(const Vector3DTemplate<RealType> &v, RealType ww) :
        x(v.x), y(v.y), z(v.z), w(ww) {}
    CUDA_COMMON_FUNCTION constexpr explicit operator Vector3DTemplate<RealType>() const {
        return Vector3DTemplate<RealType>(x, y, z);
    }

    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate operator-() const {
        return Vector4DTemplate(-x, -y, -z, -w);
    }

    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate &operator+=(const Vector4DTemplate &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate &operator-=(const Vector4DTemplate &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate &operator*=(RealType s) {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4DTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        return *this *= r;
    }

    CUDA_COMMON_FUNCTION constexpr RealType &operator[](uint32_t dim) {
        Assert(dim <= 3, "\"dim\" is out of range [0, 3].");
        return *(&x + dim);
    }
    CUDA_COMMON_FUNCTION constexpr const RealType &operator[](uint32_t dim) const {
        Assert(dim <= 3, "\"dim\" is out of range [0, 3].");
        return *(&x + dim);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc8::isnan;
        return isnan(x) || isnan(y) || isnan(z) || isnan(w);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc8::isinf;
        return isinf(x) || isinf(y) || isinf(z) || isinf(w);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc8::isfinite;
        return isfinite(x) && isfinite(y) && isfinite(z) && isfinite(w);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector4DTemplate Zero() {
        return Vector4DTemplate(0, 0, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector4DTemplate Ex() {
        return Vector4DTemplate(1, 0, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector4DTemplate Ey() {
        return Vector4DTemplate(0, 1, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector4DTemplate Ez() {
        return Vector4DTemplate(0, 0, 1, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Vector4DTemplate Ew() {
        return Vector4DTemplate(0, 0, 0, 1);
    }
};



template <typename RealType>
struct TexCoord2DTemplate {
    RealType u, v;

    CUDA_COMMON_FUNCTION TexCoord2DTemplate() {}
    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate(RealType val) :
        u(val), v(val) {}
    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate(RealType uu, RealType vv) :
        u(uu), v(vv) {}

    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate operator-() const {
        return TexCoord2DTemplate(-u, -v);
    }

    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate &operator+=(const TexCoord2DTemplate &val) {
        u += val.u;
        v += val.v;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate &operator*=(RealType s) {
        u *= s;
        v *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr TexCoord2DTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        return *this *= r;
    }

    CUDA_COMMON_FUNCTION constexpr RealType &operator[](uint32_t dim) {
        Assert(dim <= 1, "\"dim\" is out of range [0, 1].");
        return *(&u + dim);
    }
    CUDA_COMMON_FUNCTION constexpr const RealType &operator[](uint32_t dim) const {
        Assert(dim <= 1, "\"dim\" is out of range [0, 1].");
        return *(&u + dim);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc8::isnan;
        return isnan(u) || isnan(v);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc8::isinf;
        return isinf(u) || isinf(v);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc8::isfinite;
        return isfinite(u) && isfinite(v);
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr TexCoord2DTemplate Zero() {
        return TexCoord2DTemplate(0, 0);
        return TexCoord2DTemplate(0, 0);
    }
};



template <typename RealType>
struct Matrix3x3Template {
    using Vector3D = Vector3DTemplate<RealType>;

    union {
        struct {
            RealType m00, m10, m20;
        };
        Vector3D c0;
    };
    union {
        struct {
            RealType m01, m11, m21;
        };
        Vector3D c1;
    };
    union {
        struct {
            RealType m02, m12, m22;
        };
        Vector3D c2;
    };

    CUDA_COMMON_FUNCTION Matrix3x3Template() {}
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template(const RealType ar[9]) :
        m00(ar[0]), m10(ar[1]), m20(ar[2]),
        m01(ar[3]), m11(ar[4]), m21(ar[5]),
        m02(ar[6]), m12(ar[7]), m22(ar[8]) {}
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template(
        const Vector3D &col0, const Vector3D &col1, const Vector3D &col2) :
        m00(col0.x), m10(col0.y), m20(col0.z),
        m01(col1.x), m11(col1.y), m21(col1.z),
        m02(col2.x), m12(col2.y), m22(col2.z) {}

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template operator-() const {
        return Matrix3x3Template(-c0, -c1, -c2);
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &operator+=(const Matrix3x3Template &mat) {
        c0 += mat.c0;
        c1 += mat.c1;
        c2 += mat.c2;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &operator-=(const Matrix3x3Template &mat) {
        c0 -= mat.c0;
        c1 -= mat.c1;
        c2 -= mat.c2;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &operator*=(const Matrix3x3Template &mat) {
        const Vector3D r[] = { row(0), row(1), row(2) };
        c0 = Vector3D(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0));
        c1 = Vector3D(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1));
        c2 = Vector3D(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2));
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &operator*=(RealType s) {
        c0 *= s;
        c1 *= s;
        c2 *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        c0 *= r;
        c1 *= r;
        c2 *= r;
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Vector3D &operator[](uint32_t col) {
        Assert(col <= 2, "\"col\" is out of range [0, 2].");
        return *(&c0 + col);
    }
    CUDA_COMMON_FUNCTION constexpr const Vector3D &operator[](uint32_t col) const {
        Assert(col <= 2, "\"col\" is out of range [0, 2].");
        return *(&c0 + col);
    }

    CUDA_COMMON_FUNCTION constexpr const Vector3D &column(uint32_t col) const {
        Assert(col <= 2, "\"col\" is out of range [0, 2].");
        return *(&c0 + col);
    }
    CUDA_COMMON_FUNCTION constexpr Vector3D row(uint32_t r) const {
        Assert(r <= 2, "\"col\" is out of range [0, 2].");
        switch (r) {
        case 0:
            return Vector3D(m00, m01, m02);
        case 1:
            return Vector3D(m10, m11, m12);
        case 2:
            return Vector3D(m20, m21, m22);
        default:
            return Vector3D(0, 0, 0);
        }
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &swapColumns(uint32_t ca, uint32_t cb) {
        if (ca != cb) {
            Vector3D temp = column(ca);
            (*this)[ca] = (*this)[cb];
            (*this)[cb] = temp;
        }
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &swapRows(uint32_t ra, uint32_t rb) {
        if (ra != rb) {
            Vector3D temp = row(ra);
            setRow(ra, row(rb));
            setRow(rb, temp);
        }
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &setRow(uint32_t r, const Vector3D &v) {
        Assert(r <= 2, "\"r\" is out of range [0, 2].");
        c0[r] = v[0]; c1[r] = v[1]; c2[r] = v[2];
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &scaleRow(uint32_t r, RealType s) {
        Assert(r <= 2, "\"r\" is out of range [0, 2].");
        c0[r] *= s; c1[r] *= s; c2[r] *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &addRow(uint32_t r, const Vector3D &v) {
        Assert(r <= 2, "\"r\" is out of range [0, 2].");
        c0[r] += v[0]; c1[r] += v[1]; c2[r] += v[2];
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr RealType trace() const {
        return m00 + m11 + m22;
    }
    CUDA_COMMON_FUNCTION constexpr RealType determinant() const {
        return (c0[0] * (c1[1] * c2[2] - c2[1] * c1[2]) -
                c1[0] * (c0[1] * c2[2] - c2[1] * c0[2]) +
                c2[0] * (c0[1] * c1[2] - c1[1] * c0[2]));
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template& transpose() {
        _swap(m10, m01); _swap(m20, m02);
        _swap(m21, m12);
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template &invert() {
        Assert_NotImplemented();
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr bool isIdentity() const {
        return c0 == Vector3D(1, 0, 0) && c1 == Vector3D(0, 1, 0) && c2 == Vector3D(0, 0, 1);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNaN() const {
        return c0.hasNaN() || c1.hasNaN() || c2.hasNaN();
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        return c0.hasInf() || c1.hasInf() || c2.hasInf();
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        return !hasNaN() && !hasInf();
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ void decompose(Vector3D* scale, Vector3D* rotation) {
        Matrix3x3Template mat = *this;

        // JP: 拡大縮小成分
        // EN: Scale component
        *scale = Vector3D(mat.c0.length(), mat.c1.length(), mat.c2.length());

        // JP: 上記成分を排除
        // EN: Remove the above components
        if (std::fabs(scale->x) > 0)
            mat.c0 /= scale->x;
        if (std::fabs(scale->y) > 0)
            mat.c1 /= scale->y;
        if (std::fabs(scale->z) > 0)
            mat.c2 /= scale->z;

        // JP: 回転成分がXYZの順で作られている、つまりZYXp(pは何らかのベクトル)と仮定すると、行列は以下の形式をとっていると考えられる。
        //     A, B, GはそれぞれX, Y, Z軸に対する回転角度。cとsはcosとsin。
        //     cG * cB   -sG * cA + cG * sB * sA    sG * sA + cG * sB * cA
        //     sG * cB    cG * cA + sG * sB * sA   -cG * sA + sG * sB * cA
        //       -sB             cB * sA                   cB * cA
        //     したがって、3行1列成分からまずY軸に対する回転Bが求まる。
        //     次に求めたBを使って回転A, Gが求まる。数値精度を考慮すると、cBが0の場合は別の処理が必要。
        //     cBが0の場合はsBは+-1(Bが90度ならば+、-90度ならば-)なので、上の行列は以下のようになる。
        //      0   -sG * cA +- cG * sA    sG * sA +- cG * cA
        //      0    cG * cA +- sG * sA   -cG * sA +- sG * cA
        //     -+1           0                     0
        //     求めたBを使ってさらに求まる成分がないため、Aを0と仮定する。
        // EN: 
        rotation->y = std::asin(-mat.c0[2]);
        RealType cosBeta = std::cos(rotation->y);

        if (std::fabs(cosBeta) < 0.000001f) {
            rotation->x = 0;
            rotation->z = std::atan2(-mat.c1[0], mat.c1[1]);
        }
        else {
            rotation->x = std::atan2(mat.c1[2], mat.c2[2]);
            rotation->z = std::atan2(mat.c0[1], mat.c0[0]);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Matrix3x3Template Identity() {
        constexpr RealType data[] = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        };
        return Matrix3x3Template(data);
    }
};



template <typename RealType>
struct Matrix4x4Template {
    using Vector3D = Vector3DTemplate<RealType>;
    using Vector4D = Vector4DTemplate<RealType>;

    union {
        struct {
            RealType m00, m10, m20, m30;
        };
        Vector4D c0;
    };
    union {
        struct {
            RealType m01, m11, m21, m31;
        };
        Vector4D c1;
    };
    union {
        struct {
            RealType m02, m12, m22, m32;
        };
        Vector4D c2;
    };
    union {
        struct {
            RealType m03, m13, m23, m33;
        };
        Vector4D c3;
    };

    CUDA_COMMON_FUNCTION Matrix4x4Template() {}
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template(const RealType ar[16]) :
        m00(ar[ 0]), m10(ar[ 1]), m20(ar[ 2]), m30(ar[ 3]),
        m01(ar[ 4]), m11(ar[ 5]), m21(ar[ 6]), m31(ar[ 7]),
        m02(ar[ 8]), m12(ar[ 9]), m22(ar[10]), m32(ar[11]),
        m03(ar[12]), m13(ar[13]), m23(ar[14]), m33(ar[15]) {}
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template(
        const Vector4D &col0, const Vector4D &col1, const Vector4D &col2, const Vector4D &col3) :
        m00(col0.x), m10(col0.y), m20(col0.z), m30(col0.w),
        m01(col1.x), m11(col1.y), m21(col1.z), m31(col1.w),
        m02(col2.x), m12(col2.y), m22(col2.z), m32(col2.w),
        m03(col3.x), m13(col3.y), m23(col3.z), m33(col3.w) {}

    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template operator-() const {
        return Matrix4x4Template(-c0, -c1, -c2, -c3);
    }

    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &operator+=(const Matrix4x4Template &mat) {
        c0 += mat.c0;
        c1 += mat.c1;
        c2 += mat.c2;
        c3 += mat.c3;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &operator-=(const Matrix4x4Template &mat) {
        c0 -= mat.c0;
        c1 -= mat.c1;
        c2 -= mat.c2;
        c3 -= mat.c3;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &operator*=(const Matrix4x4Template &mat) {
        const Vector4D r[] = { row(0), row(1), row(2), row(3) };
        c0 = Vector4D(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
        c1 = Vector4D(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
        c2 = Vector4D(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
        c3 = Vector4D(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &operator*=(RealType s) {
        c0 *= s;
        c1 *= s;
        c2 *= s;
        c3 *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        c0 *= r;
        c1 *= r;
        c2 *= r;
        c3 *= r;
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Vector4D &operator[](uint32_t col) {
        Assert(col <= 3, "\"col\" is out of range [0, 3].");
        return *(&c0 + col);
    }
    CUDA_COMMON_FUNCTION constexpr const Vector4D &operator[](uint32_t col) const {
        Assert(col <= 3, "\"col\" is out of range [0, 3].");
        return *(&c0 + col);
    }

    CUDA_COMMON_FUNCTION constexpr const Vector4D &column(uint32_t col) const {
        Assert(col <= 3, "\"col\" is out of range [0, 3].");
        return *(&c0 + col);
    }
    CUDA_COMMON_FUNCTION constexpr Vector4D row(uint32_t r) const {
        Assert(r <= 3, "\"col\" is out of range [0, 3].");
        switch (r) {
        case 0:
            return Vector4D(m00, m01, m02, m03);
        case 1:
            return Vector4D(m10, m11, m12, m13);
        case 2:
            return Vector4D(m20, m21, m22, m23);
        case 3:
            return Vector4D(m30, m31, m32, m33);
        default:
            return Vector4D(0, 0, 0, 0);
        }
    }

    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &swapColumns(uint32_t ca, uint32_t cb) {
        if (ca != cb) {
            Vector4D temp = column(ca);
            (*this)[ca] = (*this)[cb];
            (*this)[cb] = temp;
        }
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &swapRows(uint32_t ra, uint32_t rb) {
        if (ra != rb) {
            Vector4D temp = row(ra);
            setRow(ra, row(rb));
            setRow(rb, temp);
        }
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &setRow(uint32_t r, const Vector4D &v) {
        Assert(r <= 3, "\"r\" is out of range [0, 3].");
        c0[r] = v[0];
        c1[r] = v[1];
        c2[r] = v[2];
        c3[r] = v[3];
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &scaleRow(uint32_t r, RealType s) {
        Assert(r <= 3, "\"r\" is out of range [0, 3].");
        c0[r] *= s;
        c1[r] *= s;
        c2[r] *= s;
        c3[r] *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &addRow(uint32_t r, const Vector4D &v) {
        Assert(r <= 3, "\"r\" is out of range [0, 3].");
        c0[r] += v[0];
        c1[r] += v[1];
        c2[r] += v[2];
        c3[r] += v[3];
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template &transpose() {
        _swap(m10, m01); _swap(m20, m02); _swap(m30, m03);
        _swap(m21, m12); _swap(m31, m13);
        _swap(m32, m23);
        return *this;
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ Matrix4x4Template &invert();

    CUDA_COMMON_FUNCTION constexpr bool isIdentity() const {
        return
            c0 == Vector4D(1, 0, 0, 0) &&
            c1 == Vector4D(0, 1, 0, 0) &&
            c2 == Vector4D(0, 0, 1, 0) &&
            c3 == Vector4D(0, 0, 0, 1);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNaN() const {
        return c0.hasNaN() || c1.hasNaN() || c2.hasNaN() || c3.hasNaN();
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        return c0.hasInf() || c1.hasInf() || c2.hasInf() || c3.hasInf();
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        return !hasNaN() && !hasInf();
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ void decompose(
        Vector3D* scale, Vector3D* rotation, Vector3D* translation) const {
        using Vector3D = Vector3DTemplate<RealType>;
        Matrix4x4Template<RealType> mat = *this;

        // JP: 移動成分
        // EN: Translation component
        if (translation)
            *translation = static_cast<Vector3D>(mat.c3);

        // JP: 拡大縮小成分
        // EN: Scale component
        Vector3D s(
            static_cast<Vector3D>(mat.c0).length(),
            static_cast<Vector3D>(mat.c1).length(),
            static_cast<Vector3D>(mat.c2).length());
        if (scale)
            *scale = s;

        if (!rotation)
            return;

        // JP: 上記成分を排除
        // EN: Remove the above components
        mat.c3 = Vector4DTemplate<RealType>(0, 0, 0, 1);
        if (std::fabs(s.x) > 0)
            mat.c0 /= s.x;
        if (std::fabs(s.y) > 0)
            mat.c1 /= s.y;
        if (std::fabs(s.z) > 0)
            mat.c2 /= s.z;

        // JP: 回転成分がXYZの順で作られている、つまりZYXp(pは何らかのベクトル)と仮定すると、行列は以下の形式をとっていると考えられる。
        //     A, B, GはそれぞれX, Y, Z軸に対する回転角度。cとsはcosとsin。
        //     cG * cB   -sG * cA + cG * sB * sA    sG * sA + cG * sB * cA
        //     sG * cB    cG * cA + sG * sB * sA   -cG * sA + sG * sB * cA
        //       -sB             cB * sA                   cB * cA
        //     したがって、3行1列成分からまずY軸に対する回転Bが求まる。
        //     次に求めたBを使って回転A, Gが求まる。数値精度を考慮すると、cBが0の場合は別の処理が必要。
        //     cBが0の場合はsBは+-1(Bが90度ならば+、-90度ならば-)なので、上の行列は以下のようになる。
        //      0   -sG * cA +- cG * sA    sG * sA +- cG * cA
        //      0    cG * cA +- sG * sA   -cG * sA +- sG * cA
        //     -+1           0                     0
        //     求めたBを使ってさらに求まる成分がないため、Aを0と仮定する。
        // EN: 
        rotation->y = std::asin(-mat.c0[2]);
        RealType cosBeta = std::cos(rotation->y);

        if (std::fabs(cosBeta) < 0.000001f) {
            rotation->x = 0;
            rotation->z = std::atan2(-mat.c1[0], mat.c1[1]);
        }
        else {
            rotation->x = std::atan2(mat.c1[2], mat.c2[2]);
            rotation->z = std::atan2(mat.c0[1], mat.c0[0]);
        }
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Matrix4x4Template Identity() {
        constexpr RealType data[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        };
        return Matrix4x4Template(data);
    }
};



template <typename RealType>
struct QuaternionTemplate {
    using Vector3D = Vector3DTemplate<RealType>;

    union {
        Vector3D v;
        struct {
            RealType x, y, z;
        };
    };
    RealType w;

    CUDA_COMMON_FUNCTION QuaternionTemplate() {}
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate(RealType xx, RealType yy, RealType zz, RealType ww) :
        x(xx), y(yy), z(zz), w(ww) {}
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate(const Vector3DTemplate<RealType> &v, RealType ww) :
        x(v.x), y(v.y), z(v.z), w(ww) {}
    CUDA_COMMON_FUNCTION /*constexpr*/ QuaternionTemplate(const Matrix4x4Template<RealType> &mat);

    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate operator-() const {
        return QuaternionTemplate(-v, -w);
    }

    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate &operator+=(const QuaternionTemplate &q) {
        v += q.v;
        w += q.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate &operator-=(const QuaternionTemplate &q) {
        v -= q.v;
        w -= q.w;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate &operator*=(const QuaternionTemplate &q) {
        Vector3D vv = v;
        v = cross(vv, q.v) + w * q.v + q.w * vv;
        w = w * q.w - dot(vv, q.v);
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate &operator*=(RealType s) {
        v *= s;
        w *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr QuaternionTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1) / s;
        v *= r;
        w *= r;
        return *this;
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc8::isnan;
        return v.hasNan() || isnan(w);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc8::isinf;
        return v.hasInf() || isinf(w);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc8::isfinite;
        return v.allFinite() && isfinite(w);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ RealType squaredLength() const {
        return pow2(x) + pow2(y) + pow2(z) + pow2(w);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ RealType length() const {
        return std::sqrt(squaredLength());
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ QuaternionTemplate &normalize() {
        *this /= length();
        return *this;
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template<RealType> toMatrix3x3() const {
        RealType xx = x * x, yy = y * y, zz = z * z;
        RealType xy = x * y, yz = y * z, zx = z * x;
        RealType xw = x * w, yw = y * w, zw = z * w;
        return Matrix3x3Template<RealType>(
            Vector3D(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw)),
            Vector3D(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)),
            Vector3D(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)));
    }
    CUDA_COMMON_FUNCTION constexpr Matrix4x4Template<RealType> toMatrix4x4() const {
        RealType xx = x * x, yy = y * y, zz = z * z;
        RealType xy = x * y, yz = y * z, zx = z * x;
        RealType xw = x * w, yw = y * w, zw = z * w;
        return Matrix4x4Template<RealType>(
            Vector4DTemplate<RealType>(1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (zx - yw), 0.0f),
            Vector4DTemplate<RealType>(2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw), 0.0f),
            Vector4DTemplate<RealType>(2 * (zx + yw), 2 * (yz - xw), 1 - 2 * (xx + yy), 0.0f),
            Vector4DTemplate<RealType>::Ew());
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr QuaternionTemplate Identity() {
        return QuaternionTemplate(0, 0, 0, 1);
    }
};



// ----------------------------------------------------------------
// Point3D operators and functions

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> &Point3DTemplate<RealType>::operator+=(
    const Vector3DTemplate<RealType> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> &Point3DTemplate<RealType>::operator+=(
    const Normal3DTemplate<RealType> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> &Point3DTemplate<RealType>::operator-=(
    const Vector3DTemplate<RealType> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> &Point3DTemplate<RealType>::operator-=(
    const Normal3DTemplate<RealType> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    return va.x == vb.x && va.y == vb.y && va.z == vb.z;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    return va.x != vb.x || va.y != vb.y || va.z != vb.z;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator+(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    Point3DTemplate<RealType> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator+(
    const Point3DTemplate<RealType> &va, const Vector3DTemplate<RealType> &vb) {
    Point3DTemplate<RealType> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator+(
    const Point3DTemplate<RealType> &va, const Normal3DTemplate<RealType> &vb) {
    Point3DTemplate<RealType> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> operator-(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    Vector3DTemplate<RealType> ret(va.x - vb.x, va.y - vb.y, va.z - vb.z);
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator-(
    const Point3DTemplate<RealType> &va, const Vector3DTemplate<RealType> &vb) {
    Point3DTemplate<RealType> ret = va;
    ret -= vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator-(
    const Point3DTemplate<RealType> &va, const Normal3DTemplate<RealType> &vb) {
    Point3DTemplate<RealType> ret = va;
    ret -= vb;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator*(
    const Point3DTemplate<RealType> &v, ScalarType s) {
    Point3DTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator*(
    ScalarType s, const Point3DTemplate<RealType> &v) {
    Point3DTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator/(
    const Point3DTemplate<RealType> &v, ScalarType s) {
    Point3DTemplate<RealType> ret = v;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> min(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    return Point3DTemplate<RealType>(
        std::fmin(va.x, vb.x),
        std::fmin(va.y, vb.y),
        std::fmin(va.z, vb.z));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> max(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    return Point3DTemplate<RealType>(
        std::fmax(va.x, vb.x),
        std::fmax(va.y, vb.y),
        std::fmax(va.z, vb.z));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RealType squaredDistance(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    Vector3DTemplate<RealType> vector = vb - va;
    return vector.squaredLength();
}

// END: Point3D operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// Vector3D/Normal3D operators and functions

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    return va.x == vb.x && va.y == vb.y && va.z == vb.z;
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    return va.x != vb.x || va.y != vb.y || va.z != vb.z;
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator+(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    Vector3DTemplate<RealType, isNormal> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator-(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    Vector3DTemplate<RealType, isNormal> ret = va;
    ret -= vb;
    return ret;
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator*(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    Vector3DTemplate<RealType, isNormal> ret = va;
    ret *= vb;
    return ret;
}

template <typename RealType, typename ScalarType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator*(
    const Vector3DTemplate<RealType, isNormal> &v, ScalarType s) {
    Vector3DTemplate<RealType, isNormal> ret = v;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator*(
    ScalarType s, const Vector3DTemplate<RealType, isNormal> &v) {
    Vector3DTemplate<RealType, isNormal> ret = v;
    ret *= s;
    return ret;
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator/(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    Vector3DTemplate<RealType, isNormal> ret = va;
    ret /= vb;
    return ret;
}

template <typename RealType, typename ScalarType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> operator/(
    const Vector3DTemplate<RealType, isNormal> &v, ScalarType s) {
    Vector3DTemplate<RealType, isNormal> ret = v;
    ret /= s;
    return ret;
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Vector3DTemplate<RealType, isNormal> normalize(
    const Vector3DTemplate<RealType, isNormal> &v) {
    RealType l = v.length();
    return v / l;
}

template <typename RealType, bool aIsNormal, bool bIsNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RealType dot(
    const Vector3DTemplate<RealType, aIsNormal> &va, const Vector3DTemplate<RealType, bIsNormal> &vb) {
    return va.x * vb.x + va.y * vb.y + va.z * vb.z;
}

template <typename RealType, bool aIsNormal, bool bIsNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ RealType absDot(
    const Vector3DTemplate<RealType, aIsNormal> &va, const Vector3DTemplate<RealType, bIsNormal> &vb) {
    return std::fabs(dot(va, vb));
}

template <typename RealType, bool aIsNormal, bool bIsNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, aIsNormal> cross(
    const Vector3DTemplate<RealType, aIsNormal> &va, const Vector3DTemplate<RealType, bIsNormal> &vb) {
    return Vector3DTemplate<RealType, aIsNormal>(
        va.y * vb.z - va.z * vb.y,
        va.z * vb.x - va.x * vb.z,
        va.x * vb.y - va.y * vb.x);
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> min(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    return Vector3DTemplate<RealType, isNormal>(
        std::fmin(va.x, vb.x),
        std::fmin(va.y, vb.y),
        std::fmin(va.z, vb.z));
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType, isNormal> max(
    const Vector3DTemplate<RealType, isNormal> &va, const Vector3DTemplate<RealType, isNormal> &vb) {
    return Vector3DTemplate<RealType, isNormal>(
        std::fmax(va.x, vb.x),
        std::fmax(va.y, vb.y),
        std::fmax(va.z, vb.z));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Vector3DTemplate<RealType> halfVector(
    const Vector3DTemplate<RealType> &va, const Vector3DTemplate<RealType> &vb) {
    return normalize(va + vb);
}

template <typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> safeDivide(
    const Vector3DTemplate<RealType> &a, const Vector3DTemplate<RealType> &b) {
    RealType zero = static_cast<RealType>(0);
    return Vector3DTemplate<RealType>(
        b.x != 0 ? a.x / b.x : zero,
        b.y != 0 ? a.y / b.y : zero,
        b.z != 0 ? a.z / b.z : zero);
}

// END: Vector3D/Normal3D operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// Vector4D operators and functions

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    return va.x == vb.x && va.y == vb.y && va.z == vb.z && va.w == vb.w;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    return va.x != vb.x || va.y != vb.y || va.z != vb.z || va.w != vb.w;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> operator+(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    Vector4DTemplate<RealType> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> operator-(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    Vector4DTemplate<RealType> ret = va;
    ret -= vb;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> operator*(
    const Vector4DTemplate<RealType> &v, ScalarType s) {
    Vector4DTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> operator*(
    ScalarType s, const Vector4DTemplate<RealType> &v) {
    Vector4DTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> operator/(
    const Vector4DTemplate<RealType> &v, RealType s) {
    Vector4DTemplate<RealType> ret = v;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RealType dot(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> min(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    return Vector4DTemplate<RealType>(
        std::fmin(va.x, vb.x),
        std::fmin(va.y, vb.y),
        std::fmin(va.z, vb.z),
        std::fmin(va.w, vb.w));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> max(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    return Vector4DTemplate<RealType>(
        std::fmax(va.x, vb.x),
        std::fmax(va.y, vb.y),
        std::fmax(va.z, vb.z),
        std::fmax(va.w, vb.w));
}

// END: Vector4D operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// TexCoord2D operators and functions

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const TexCoord2DTemplate<RealType> &tca, const TexCoord2DTemplate<RealType> &tcb) {
    return tca.u == tcb.u && tca.v == tcb.v;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const TexCoord2DTemplate<RealType> &tca, const TexCoord2DTemplate<RealType> &tcb) {
    return tca.u != tcb.u || tca.v != tcb.v;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TexCoord2DTemplate<RealType> operator+(
    const TexCoord2DTemplate<RealType> &tca, const TexCoord2DTemplate<RealType> &tcb) {
    TexCoord2DTemplate<RealType> ret = tca;
    ret += tcb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> operator-(
    const TexCoord2DTemplate<RealType> &tca, const TexCoord2DTemplate<RealType> &tcb) {
    Vector3DTemplate<RealType> ret(tca.u - tcb.u, tca.v - tcb.v, tca.z - tcb.z);
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TexCoord2DTemplate<RealType> operator*(
    const TexCoord2DTemplate<RealType> &tc, ScalarType s) {
    TexCoord2DTemplate<RealType> ret = tc;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TexCoord2DTemplate<RealType> operator*(
    ScalarType s, const TexCoord2DTemplate<RealType> &tc) {
    TexCoord2DTemplate<RealType> ret = tc;
    ret *= s;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TexCoord2DTemplate<RealType> operator/(
    const TexCoord2DTemplate<RealType> &tc, ScalarType s) {
    TexCoord2DTemplate<RealType> ret = tc;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TexCoord2DTemplate<RealType> min(
    const TexCoord2DTemplate<RealType> &tca, const TexCoord2DTemplate<RealType> &tcb) {
    using rtc8::min;
    return TexCoord2DTemplate<RealType>(min(tca.u, tcb.u), min(tca.v, tcb.v), min(tca.z, tcb.z));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TexCoord2DTemplate<RealType> max(
    const TexCoord2DTemplate<RealType> &tca, const TexCoord2DTemplate<RealType> &tcb) {
    using rtc8::max;
    return TexCoord2DTemplate<RealType>(max(tca.u, tcb.u), max(tca.v, tcb.v), max(tca.z, tcb.z));
}

// END: TexCoord2D operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// Matrix3x3 operators and functions

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> operator+(
    const Matrix3x3Template<RealType> &matA, const Matrix3x3Template<RealType> &matB) {
    Matrix3x3Template<RealType> ret = matA;
    ret += matB;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> operator-(
    const Matrix3x3Template<RealType> &matA, const Matrix3x3Template<RealType> &matB) {
    Matrix3x3Template<RealType> ret = matA;
    ret -= matB;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> operator*(
    const Matrix3x3Template<RealType> &matA, const Matrix3x3Template<RealType> &matB) {
    Matrix3x3Template<RealType> ret = matA;
    ret *= matB;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator*(
    const Matrix3x3Template<RealType> &mat, const Point3DTemplate<RealType> &p) {
    Vector3DTemplate<RealType> v(p);
    return Point3DTemplate<RealType>(dot(mat.row(0), v), dot(mat.row(1), v), dot(mat.row(2), v));
}

template <typename RealType, bool isNormal>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> operator*(
    const Matrix3x3Template<RealType> &mat, const Vector3DTemplate<RealType, isNormal> &v) {
    return Vector3DTemplate<RealType, isNormal>(dot(mat.row(0), v), dot(mat.row(1), v), dot(mat.row(2), v));
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> operator*(
    const Matrix3x3Template<RealType> &mat, ScalarType s) {
    Matrix3x3Template<RealType> ret = mat;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> operator*(
    const Matrix3x3Template<RealType> &mat, ScalarType s) {
    Matrix3x3Template<RealType> ret = mat;
    ret *= s;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> operator/(
    const Matrix3x3Template<RealType> &mat, ScalarType s) {
    Matrix3x3Template<RealType> ret = mat;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> transpose(
    const Matrix3x3Template<RealType> &mat) {
    Matrix3x3Template<RealType> ret = mat;
    ret.transpose();
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> invert(
    const Matrix3x3Template<RealType> &mat) {
    Matrix3x3Template<RealType> ret = mat;
    ret.invert();
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> scale3x3(
    RealType sx, RealType sy, RealType sz) {
    return Matrix3x3Template<RealType>(
        sx * Vector3DTemplate<RealType>::Ex(),
        sy * Vector3DTemplate<RealType>::Ey(),
        sz * Vector3DTemplate<RealType>::Ez());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> scale3x3(
    const Vector3DTemplate<RealType> &s) {
    return scale3x3(s.x, s.y, s.z);
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix3x3Template<RealType> scale3x3(
    RealType s) {
    return scale3x3(Vector3DTemplate<RealType>(s, s, s));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix3x3Template<RealType> rotate3x3(
    RealType angle, const Vector3DTemplate<RealType> &axis) {
    Matrix3x3Template<RealType> matrix;
    Vector3DTemplate<RealType> nAxis = normalize(axis);
    RealType s, c;
    sincos(angle, &s, &c);
    RealType oneMinusC = 1 - c;

    matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
    matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
    matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
    matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
    matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
    matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
    matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
    matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
    matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

    return matrix;
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix3x3Template<RealType> rotate3x3(
    RealType angle, RealType ax, RealType ay, RealType az) {
    return rotate3x3(angle, Vector3DTemplate<RealType>(ax, ay, az));
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix3x3Template<RealType> rotateX3x3(
    RealType angle) {
    return rotate3x3(angle, Vector3DTemplate<RealType>::Ex());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix3x3Template<RealType> rotateY3x3(
    RealType angle) {
    return rotate3x3(angle, Vector3DTemplate<RealType>::Ey());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix3x3Template<RealType> rotateZ3x3(
    RealType angle) {
    return rotate3x3(angle, Vector3DTemplate<RealType>::Ez());
}

// END: Matrix3x3 operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// Matrix4x4 operators and functions

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/
Matrix4x4Template<RealType> &Matrix4x4Template<RealType>::invert() {
    bool colDone[] = { false, false, false, false };
    struct SwapPair {
        int a, b;
        CUDA_COMMON_FUNCTION constexpr SwapPair(int aa, int bb) : a(aa), b(bb) {}
    };
    SwapPair swapPairs[] = { SwapPair(0, 0), SwapPair(0, 0), SwapPair(0, 0), SwapPair(0, 0) };
    for (int pass = 0; pass < 4; ++pass) {
        int pvCol = 0;
        int pvRow = 0;
        RealType maxPivot = -1;
        for (int c = 0; c < 4; ++c) {
            if (colDone[c])
                continue;
            for (int r = 0; r < 4; ++r) {
                if (colDone[r])
                    continue;

                RealType absValue = std::fabs((*this)[c][r]);
                if (absValue > maxPivot) {
                    pvCol = c;
                    pvRow = r;
                    maxPivot = absValue;
                }
            }
        }

        swapRows(pvRow, pvCol);
        swapPairs[pass] = SwapPair(pvRow, pvCol);

        RealType pivot = (*this)[pvCol][pvCol];
        if (pivot == 0) {
            Vector4DTemplate<RealType> nanVec(NAN);
            *this = Matrix4x4Template<RealType>(nanVec, nanVec, nanVec, nanVec);
            return *this;
        }

        (*this)[pvCol][pvCol] = 1;
        scaleRow(pvCol, 1 / pivot);
        Vector4DTemplate<RealType> addendRow = row(pvCol);
        for (int r = 0; r < 4; ++r) {
            if (r != pvCol) {
                RealType s = (*this)[pvCol][r];
                (*this)[pvCol][r] = 0;
                addRow(r, -s * addendRow);
            }
        }

        colDone[pvCol] = true;
    }

    for (int pass = 3; pass >= 0; --pass) {
        const SwapPair &pair = swapPairs[pass];
        swapColumns(pair.a, pair.b);
    }

    return *this;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> operator+(
    const Matrix4x4Template<RealType> &matA, const Matrix4x4Template<RealType> &matB) {
    Matrix4x4Template<RealType> ret = matA;
    ret += matB;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> operator-(
    const Matrix4x4Template<RealType> &matA, const Matrix4x4Template<RealType> &matB) {
    Matrix4x4Template<RealType> ret = matA;
    ret -= matB;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> operator*(
    const Matrix4x4Template<RealType> &matA, const Matrix4x4Template<RealType> &matB) {
    Matrix4x4Template<RealType> ret = matA;
    ret *= matB;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> operator*(
    const Matrix4x4Template<RealType> &mat, const Vector3DTemplate<RealType> &v) {
    using Vector3D = Vector3DTemplate<RealType>;
    return Vector3D(
        dot(static_cast<Vector3D>(mat.row(0)), v),
        dot(static_cast<Vector3D>(mat.row(1)), v),
        dot(static_cast<Vector3D>(mat.row(2)), v));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> operator*(
    const Matrix4x4Template<RealType> &mat, const Vector4DTemplate<RealType> &v) {
    return Vector4DTemplate<RealType>(
        dot(mat.row(0), v),
        dot(mat.row(1), v),
        dot(mat.row(2), v),
        dot(mat.row(3), v));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator*(
    const Matrix4x4Template<RealType> &mat, const Point3DTemplate<RealType> &p) {
    Vector4DTemplate<RealType> ph(p.x, p.y, p.z, 1);
    Vector4DTemplate<RealType> pht = mat * ph;
    return Point3DTemplate<RealType>(pht.x, pht.y, pht.z);
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> operator*(
    const Matrix4x4Template<RealType> &mat, ScalarType s) {
    Matrix4x4Template<RealType> ret = mat;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> operator*(
    const Matrix4x4Template<RealType> &mat, ScalarType s) {
    Matrix4x4Template<RealType> ret = mat;
    ret *= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> operator/(
    const Matrix4x4Template<RealType> &mat, RealType s) {
    Matrix4x4Template<RealType> ret = mat;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> transpose(
    const Matrix4x4Template<RealType> &mat) {
    Matrix4x4Template<RealType> ret = mat;
    ret.transpose();
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> invert(
    const Matrix4x4Template<RealType> &mat) {
    Matrix4x4Template<RealType> ret = mat;
    ret.invert();
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> scale4x4(
    const Vector3DTemplate<RealType> &s) {
    return Matrix4x4Template<RealType>(
        s.x * Vector4DTemplate<RealType>::Ex(),
        s.y * Vector4DTemplate<RealType>::Ey(),
        s.z * Vector4DTemplate<RealType>::Ez(),
        Vector4DTemplate<RealType>::Ew());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> scale4x4(
    RealType sx, RealType sy, RealType sz) {
    return scale4x4(Vector3DTemplate<RealType>(sx, sy, sz));
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> scale4x4(
    RealType s) {
    return scale4x4(Vector3DTemplate<RealType>(s, s, s));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> translate4x4(
    RealType tx, RealType ty, RealType tz) {
    return Matrix4x4Template<RealType>(
        Vector4DTemplate<RealType>::Ex(),
        Vector4DTemplate<RealType>::Ey(),
        Vector4DTemplate<RealType>::Ez(),
        Vector4DTemplate<RealType>(tx, ty, tz, 1));
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> translate4x4(
    const Vector3DTemplate<RealType> &t) {
    return translate4x4(t.x, t.y, t.z);
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> translate4x4(
    const Point3DTemplate<RealType> &t) {
    return translate4x4(t.x, t.y, t.z);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> rotate4x4(
    RealType angle, const Vector3DTemplate<RealType> &axis) {
    Matrix4x4Template<RealType> matrix;
    Vector3DTemplate<RealType> nAxis = normalize(axis);
    RealType s, c;
    sincos(angle, &s, &c);
    RealType oneMinusC = 1 - c;

    matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
    matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
    matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
    matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
    matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
    matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
    matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
    matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
    matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

    matrix.m30 = matrix.m31 = matrix.m32 = matrix.m03 = matrix.m13 = matrix.m23 = 0;
    matrix.m33 = 1;

    return matrix;
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> rotate4x4(
    RealType angle, RealType ax, RealType ay, RealType az) {
    return rotate4x4(angle, Vector3DTemplate<RealType>(ax, ay, az));
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> rotateX4x4(
    RealType angle) {
    return rotate4x4(angle, Vector3DTemplate<RealType>::Ex());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> rotateY4x4(
    RealType angle) {
    return rotate4x4(angle, Vector3DTemplate<RealType>::Ey());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> rotateZ4x4(
    RealType angle) {
    return rotate4x4(angle, Vector3DTemplate<RealType>::Ez());
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Matrix4x4Template<RealType> lookAt(
    const Point3DTemplate<RealType> &eye, const Point3DTemplate<RealType> &tgt,
    const Vector3DTemplate<RealType> &up) {
    using Vector3D = Vector3DTemplate<RealType>;
    using Vector4D = Vector4DTemplate<RealType>;
    Vector3D z = normalize(eye - tgt);
    Vector3D x = normalize(cross(up, z));
    Vector3D y = cross(z, x);
    Vector4D t = Vector4D(-dot(Vector3D(eye), x),
                          -dot(Vector3D(eye), y),
                          -dot(Vector3D(eye), z), 1);

    return Matrix4x4Template<RealType>(Vector4D(x.x, y.x, z.x, 0),
                                       Vector4D(x.y, y.y, z.y, 0),
                                       Vector4D(x.z, y.z, z.z, 0),
                                       t);
}

// END: Matrix4x4 operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// Quaternion operators and functions

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType>::QuaternionTemplate(
    const Matrix4x4Template<RealType> &mat) {
    RealType trace = mat[0][0] + mat[1][1] + mat[2][2];
    if (trace > 0) {
        RealType s = std::sqrt(trace + 1);
        v = (static_cast<RealType>(0.5) / s) *
            Vector3D(mat[1][2] - mat[2][1], mat[2][0] - mat[0][2], mat[0][1] - mat[1][0]);
        w = s / 2;
    }
    else {
        const int nxt[3] = { 1, 2, 0 };
        RealType q[3];
        int i = 0;
        if (mat[1][1] > mat[0][0])
            i = 1;
        if (mat[2][2] > mat[i][i])
            i = 2;
        int j = nxt[i];
        int k = nxt[j];
        RealType s = std::sqrt((mat[i][i] - (mat[j][j] + mat[k][k])) + 1);
        q[i] = s * 0;
        if (s != 0)
            s = static_cast<RealType>(0.5) / s;
        w = (mat[j][k] - mat[k][j]) * s;
        q[j] = (mat[i][j] + mat[j][i]) * s;
        q[k] = (mat[i][k] + mat[k][i]) * s;
        v = Vector3D(q[0], q[1], q[2]);
    }
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    return qa.v == qb.v && qa.w == qb.w;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    return qa.v != qb.v || qa.w != qb.w;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> operator+(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    QuaternionTemplate<RealType> ret = qa;
    ret += qb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> operator-(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    QuaternionTemplate<RealType> ret = qa;
    ret -= qb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> operator*(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    QuaternionTemplate<RealType> ret = qa;
    ret *= qb;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> operator*(
    const QuaternionTemplate<RealType> &q, ScalarType s) {
    QuaternionTemplate<RealType> ret = q;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> operator*(
    ScalarType s, const QuaternionTemplate<RealType> &q) {
    QuaternionTemplate<RealType> ret = q;
    ret *= s;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> operator/(
    const QuaternionTemplate<RealType> &q, ScalarType s) {
    QuaternionTemplate<RealType> ret = q;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RealType dot(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    return dot(qa.v, qb.v) + qa.w * qb.w;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr QuaternionTemplate<RealType> conjugate(
    const QuaternionTemplate<RealType> &q) {
    return QuaternionTemplate<RealType>(-q.v, q.w);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> normalize(
    const QuaternionTemplate<RealType> &q) {
    QuaternionTemplate<RealType> ret = q;
    ret.normalize();
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotate(
    RealType angle, const Vector3DTemplate<RealType> &axis) {
    RealType s, c;
    sincos(angle / 2, &s, &c);
    return QuaternionTemplate<RealType>(s * normalize(axis), c);
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotate(
    RealType angle, RealType ax, RealType ay, RealType az) {
    return qRotate(angle, Vector3DTemplate<RealType>(ax, ay, az));
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotateX(
    RealType angle) {
    return qRotate(angle, Vector3DTemplate<RealType>::Ex());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotateY(
    RealType angle) {
    return qRotate(angle, Vector3DTemplate<RealType>::Ey());
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotateZ(
    RealType angle) {
    return qRotate(angle, Vector3DTemplate<RealType>::Ez());
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qLookAt(
    const Vector3DTemplate<RealType> &dir, const Vector3DTemplate<RealType> &up) {
    using Quaternion = QuaternionTemplate<RealType>;
    using Vector3D = Vector3DTemplate<RealType>;
    Vector3D nD = normalize(-dir);
    Vector3D nS = normalize(cross(up, nD));
    Vector3D nU = cross(nD, nS);
    RealType trace = nS.x + nU.y + nD.z;
    Quaternion ret;
    if (trace > 0) {
        RealType s = static_cast<RealType>(0.5) / std::sqrt(trace + 1);
        ret.w = static_cast<RealType>(0.25) / s;
        ret.x = (nU.z - nD.y) * s;
        ret.y = (nD.x - nS.z) * s;
        ret.z = (nS.y - nU.x) * s;
    }
    else {
        if (nS.x > nU.y && nS.x > nD.z) {
            RealType s = 2 * std::sqrt(1 + nS.x - nU.y - nD.z);
            ret.w = (nU.z - nD.y) / s;
            ret.x = static_cast<RealType>(0.25) * s;
            ret.y = (nU.x + nS.y) / s;
            ret.z = (nD.x + nS.z) / s;
        }
        else if (nU.y > nD.z) {
            RealType s = 2 * std::sqrt(1 + nU.y - nS.x - nD.z);
            ret.w = (nD.x - nS.z) / s;
            ret.x = (nU.x + nS.y) / s;
            ret.y = static_cast<RealType>(0.25) * s;
            ret.z = (nD.y + nU.z) / s;
        }
        else {
            RealType s = 2 * std::sqrt(1 + nD.z - nS.x - nU.y);
            ret.w = (nS.y - nU.x) / s;
            ret.x = (nD.x + nS.z) / s;
            ret.y = (nD.y + nU.z) / s;
            ret.z = static_cast<RealType>(0.25) * s;
        }
    }

    // lookAt matrix is applied to objects instead of camera.
    ret = conjugate(ret);

    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> slerp(
    const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb, RealType t,
    bool shorterPath = false) {
    RealType cosTheta = dot(qa, qb);
    if (cosTheta > static_cast<RealType>(0.9995))
        return normalize((1 - t) * qa + t * qb);
    else {
        QuaternionTemplate<RealType> qbs = qb;
        if (shorterPath && cosTheta < 0) {
            qbs = -qb;
            cosTheta = -cosTheta;
        }
        RealType theta = std::acos(clamp(cosTheta, static_cast<RealType>(-1), static_cast<RealType>(1)));
        RealType thetap = theta * t;
        QuaternionTemplate<RealType> qPerp = normalize(qbs - qa * cosTheta);
        RealType sinThetaP, cosThetaP;
        sincos(thetap, &sinThetaP, &cosThetaP);
        return qa * cosThetaP + qPerp * sinThetaP;
    }
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ void decompose(
    const Matrix4x4Template<RealType> &mat,
    Vector3DTemplate<RealType>* T,
    QuaternionTemplate<RealType>* R,
    Matrix4x4Template<RealType>* S) {
    T->x = mat[3][0];
    T->y = mat[3][1];
    T->z = mat[3][2];

    Matrix4x4Template<RealType> matRS = mat;
    for (int i = 0; i < 3; ++i)
        matRS[3][i] = matRS[i][3] = 0;
    matRS[3][3] = 1;

    RealType norm;
    int count = 0;
    Matrix4x4Template<RealType> curR = matRS;
    do {
        Matrix4x4Template<RealType> itR = invert(transpose(curR));
        Matrix4x4Template<RealType> nextR = static_cast<RealType>(0.5) * (curR + itR);

        norm = 0;
        for (int i = 0; i < 3; ++i) {
            using std::fabs;
            RealType n =
                std::fabs(curR[0][i] - nextR[0][i]) +
                std::fabs(curR[1][i] - nextR[1][i]) +
                std::fabs(curR[2][i] - nextR[2][i]);
            norm = std::fmax(norm, n);
        }
        curR = nextR;
    } while (++count < 100 && norm > static_cast<RealType>(0.0001));
    *R = QuaternionTemplate<RealType>(curR);

    *S = invert(curR) * matRS;
}

// END: Quaternion operators and functions
// ----------------------------------------------------------------



template <typename RealType>
struct BoundingBox3DTemplate {
    using PointType = Point3DTemplate<RealType>;

    PointType minP, maxP;

    CUDA_COMMON_FUNCTION constexpr BoundingBox3DTemplate() : minP(INFINITY), maxP(-INFINITY) {}
    CUDA_COMMON_FUNCTION constexpr BoundingBox3DTemplate(const PointType &p) :
        minP(p), maxP(p) {}
    CUDA_COMMON_FUNCTION constexpr BoundingBox3DTemplate(
        const PointType &_minP, const PointType &_maxP) :
        minP(_minP), maxP(_maxP) {}

    CUDA_COMMON_FUNCTION constexpr PointType calcCentroid() const {
        return static_cast<RealType>(0.5) * (minP + maxP);
    }

    CUDA_COMMON_FUNCTION constexpr RealType calcHalfSurfaceArea() const {
        Vector3DTemplate<RealType> d = maxP - minP;
        return d.x * d.y + d.y * d.z + d.z * d.x;
    }

    CUDA_COMMON_FUNCTION constexpr RealType calcVolume() const {
        Vector3DTemplate<RealType> d = maxP - minP;
        return d.x * d.y * d.z;
    }

    CUDA_COMMON_FUNCTION constexpr RealType calcCenterOfAxis(uint32_t dim) const {
        return (minP[dim] + maxP[dim]) * static_cast<RealType>(0.5);
    }

    CUDA_COMMON_FUNCTION constexpr RealType calcWidth(uint32_t dim) const {
        return maxP[dim] - minP[dim];
    }

    CUDA_COMMON_FUNCTION constexpr uint32_t calcWidestDimension() const {
        Vector3DTemplate<RealType> d = maxP - minP;
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    CUDA_COMMON_FUNCTION constexpr bool isValid() const {
        Vector3DTemplate<RealType> d = maxP - minP;
        return d.x >= 0 && d.y >= 0 && d.z >= 0;
    }

    CUDA_COMMON_FUNCTION BoundingBox3DTemplate constexpr &unify(const PointType &p) {
        minP = min(minP, p);
        maxP = max(maxP, p);
        return *this;
    }

    CUDA_COMMON_FUNCTION BoundingBox3DTemplate constexpr &unify(const BoundingBox3DTemplate &b) {
        minP = min(minP, b.minP);
        maxP = max(maxP, b.maxP);
        return *this;
    }

    CUDA_COMMON_FUNCTION BoundingBox3DTemplate constexpr &intersect(const BoundingBox3DTemplate &b) {
        minP = max(minP, b.minP);
        maxP = min(maxP, b.maxP);
        return *this;
    }

    CUDA_COMMON_FUNCTION bool constexpr contains(const PointType &p) const {
        return ((p.x >= minP.x && p.x < maxP.x) &&
                (p.y >= minP.y && p.y < maxP.y) &&
                (p.z >= minP.z && p.z < maxP.z));
    }

    CUDA_COMMON_FUNCTION constexpr PointType calcLocalCoordinates(
        const PointType &p) const {
        return static_cast<PointType>(safeDivide(p - minP, maxP - minP));
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNaN() const {
        return minP.hasNaN() || maxP.hasNaN();
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        return minP.hasInf() || maxP.hasInf();
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        return !hasNaN() && !hasInf();
    }
};

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE BoundingBox3DTemplate<RealType> constexpr unify(
    const BoundingBox3DTemplate<RealType> &bb, const Point3DTemplate<RealType> &p) {
    BoundingBox3DTemplate<RealType> ret = bb;
    ret.unify(p);
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE BoundingBox3DTemplate<RealType> constexpr unify(
    const BoundingBox3DTemplate<RealType> &bbA, const BoundingBox3DTemplate<RealType> &bbB) {
    BoundingBox3DTemplate<RealType> ret = bbA;
    ret.unify(bbB);
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE BoundingBox3DTemplate<RealType> constexpr intersect(
    const BoundingBox3DTemplate<RealType> &bbA, const BoundingBox3DTemplate<RealType> &bbB) {
    BoundingBox3DTemplate<RealType> ret = bbA;
    ret.intersect(bbB);
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr BoundingBox3DTemplate<RealType> operator*(
    const Matrix4x4Template<RealType> &mat, const BoundingBox3DTemplate<RealType> &bb) {
    BoundingBox3DTemplate ret(Point3DTemplate<RealType>(INFINITY), Point3DTemplate<RealType>(-INFINITY));
    ret.unify(mat * Point3DTemplate<RealType>(bb.minP.x, bb.minP.y, bb.minP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.maxP.x, bb.minP.y, bb.minP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.minP.x, bb.maxP.y, bb.minP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.maxP.x, bb.maxP.y, bb.minP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.minP.x, bb.minP.y, bb.maxP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.maxP.x, bb.minP.y, bb.maxP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.minP.x, bb.maxP.y, bb.maxP.z));
    ret.unify(mat * Point3DTemplate<RealType>(bb.maxP.x, bb.maxP.y, bb.maxP.z));
    return ret;
}



template <typename RealType>
struct RGBTemplate {
    RealType r, g, b;

    CUDA_COMMON_FUNCTION RGBTemplate() {}
    CUDA_COMMON_FUNCTION constexpr RGBTemplate(RealType v) :
        r(v), g(v), b(v) {}
    CUDA_COMMON_FUNCTION constexpr RGBTemplate(RealType rr, RealType gg, RealType bb) :
        r(rr), g(gg), b(bb) {}

    CUDA_COMMON_FUNCTION constexpr RGBTemplate operator+() const {
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBTemplate operator-() const {
        return RGBTemplate(-r, -g, -b);
    }

    CUDA_COMMON_FUNCTION constexpr RGBTemplate &operator+=(const RGBTemplate &v) {
        r += v.r;
        g += v.g;
        b += v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBTemplate &operator-=(const RGBTemplate &v) {
        r -= v.r;
        g -= v.g;
        b -= v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBTemplate &operator*=(const RGBTemplate &v) {
        r *= v.r;
        g *= v.g;
        b *= v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBTemplate &operator*=(RealType s) {
        r *= s;
        g *= s;
        b *= s;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBTemplate &operator/=(const RGBTemplate &v) {
        r /= v.r;
        g /= v.g;
        b /= v.b;
        return *this;
    }
    CUDA_COMMON_FUNCTION constexpr RGBTemplate &operator/=(RealType s) {
        RealType r = static_cast<RealType>(1.0) / s;
        return *this *= r;
    }

    CUDA_COMMON_FUNCTION constexpr RealType &operator[](uint32_t ch) {
        Assert(ch <= 2, "\"ch\" is out of range [0, 2].");
        return *(&r + ch);
    }
    CUDA_COMMON_FUNCTION constexpr const RealType &operator[](uint32_t ch) const {
        Assert(ch <= 2, "\"ch\" is out of range [0, 2].");
        return *(&r + ch);
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasNan() const {
        using rtc8::isnan;
        return isnan(r) || isnan(g) || isnan(b);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool hasInf() const {
        using rtc8::isinf;
        return isinf(r) || isinf(g) || isinf(b);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allFinite() const {
        using rtc8::isfinite;
        return isfinite(r) && isfinite(g) && isfinite(b);
    }
    CUDA_COMMON_FUNCTION constexpr bool hasNonZero() const {
        return r != 0 || g != 0 || b != 0;
    }
    CUDA_COMMON_FUNCTION constexpr bool hasNegative() const {
        return r < 0 || g < 0 || b < 0;
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ bool allNonNegativeFinite() const {
        return !hasNegative() && allFinite();
    }

    CUDA_COMMON_FUNCTION constexpr float min() const {
        return std::fmin(r, std::fmin(g, b));
    }
    CUDA_COMMON_FUNCTION constexpr float max() const {
        return std::fmax(r, std::fmax(g, b));
    }
    CUDA_COMMON_FUNCTION constexpr float luminance() const {
        return
            static_cast<RealType>(0.2126729) * r +
            static_cast<RealType>(0.7151522) * g +
            static_cast<RealType>(0.0721750) * b;
    }

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBTemplate Zero() {
        return RGBTemplate(0, 0, 0);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBTemplate One() {
        return RGBTemplate(1, 1, 1);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBTemplate Infinity() {
        return RGBTemplate(INFINITY, INFINITY, INFINITY);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBTemplate NaN() {
        return RGBTemplate(NAN, NAN, NAN);
    }
    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr RGBTemplate fromXYZ(RealType xyz[3]) {
        RGBTemplate ret(
            static_cast<RealType>(3.2404542) * xyz[0] +
            static_cast<RealType>(-1.5371385) * xyz[1] +
            static_cast<RealType>(-0.4985314) * xyz[2],
            static_cast<RealType>(-0.9692660) * xyz[0] +
            static_cast<RealType>(1.8760108) * xyz[1] +
            static_cast<RealType>(0.0415560) * xyz[2],
            static_cast<RealType>(0.0556434) * xyz[0] +
            static_cast<RealType>(-0.2040259) * xyz[1] +
            static_cast<RealType>(1.0572252) * xyz[2]);
        return ret;
    }
};

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    return va.r == vb.r && va.g == vb.g && va.b == vb.b;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    return va.r != vb.r || va.g != vb.g || va.b != vb.b;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> operator+(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    RGBTemplate<RealType> ret = va;
    ret += vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> operator-(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    RGBTemplate<RealType> ret = va;
    ret -= vb;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> operator*(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    RGBTemplate<RealType> ret = va;
    ret *= vb;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> operator*(
    const RGBTemplate<RealType> &v, ScalarType s) {
    RGBTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename ScalarType, typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> operator*(
    ScalarType s, const RGBTemplate<RealType> &v) {
    RGBTemplate<RealType> ret = v;
    ret *= s;
    return ret;
}

template <typename RealType, typename ScalarType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> operator/(
    const RGBTemplate<RealType> &v, ScalarType s) {
    RGBTemplate<RealType> ret = v;
    ret /= s;
    return ret;
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> min(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    return RGBTemplate<RealType>(
        std::fmin(va.r, vb.r),
        std::fmin(va.g, vb.g),
        std::fmin(va.b, vb.b));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> max(
    const RGBTemplate<RealType> &va, const RGBTemplate<RealType> &vb) {
    return RGBTemplate<RealType>(
        std::fmax(va.r, vb.r),
        std::fmax(va.g, vb.g),
        std::fmax(va.b, vb.b));
}

template <typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> safeDivide(
    const RGBTemplate<RealType> &a, const RGBTemplate<RealType> &b) {
    RealType zero = static_cast<RealType>(0);
    return RGBTemplate<RealType>(
        b.r != 0 ? a.r / b.r : zero,
        b.g != 0 ? a.g / b.g : zero,
        b.b != 0 ? a.b / b.b : zero);
}

template <typename RealType>
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr RGBTemplate<RealType> safeDivide(
    const RGBTemplate<RealType> &a, RealType b) {
    RealType zero = static_cast<RealType>(0);
    return RGBTemplate<RealType>(
        b != 0 ? a.r / b : zero,
        b != 0 ? a.g / b : zero,
        b != 0 ? a.b / b : zero);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RGBTemplate<RealType> HSVtoRGB(RealType h, RealType s, RealType v) {
    if (s == 0)
        return RGBTemplate<RealType>(v, v, v);

    h = h - std::floor(h);
    int32_t hi = static_cast<int32_t>(h * 6);
    RealType f = h * 6 - hi;
    RealType m = v * (1 - s);
    RealType n = v * (1 - s * f);
    RealType k = v * (1 - s * (1 - f));
    if (hi == 0)
        return RGBTemplate<RealType>(v, k, m);
    else if (hi == 1)
        return RGBTemplate<RealType>(n, v, m);
    else if (hi == 2)
        return RGBTemplate<RealType>(m, v, k);
    else if (hi == 3)
        return RGBTemplate<RealType>(m, n, v);
    else if (hi == 4)
        return RGBTemplate<RealType>(k, m, v);
    else if (hi == 5)
        return RGBTemplate<RealType>(v, m, n);
    return RGBTemplate<RealType>(0, 0, 0);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE void RGBtoHSV(
    RGBTemplate<RealType> &input, RealType* h, RealType* s, RealType* v) {
    RealType minV = min(min(input.r, input.g), input.b);
    RealType maxV = max(max(input.r, input.g), input.b);

    *v = maxV;
    RealType delta = maxV - minV;
    if (delta < 1e-5f || maxV == 0.0f) {
        *h = 0.0f;
        *s = 0.0f;
        return;
    }

    *s = delta / maxV;

    if (input.r >= maxV)
        *h = (input.g - input.b) / delta;
    else if (input.g >= maxV)
        *h = 2.0f + (input.b - input.r) / delta;
    else
        *h = 4.0f + (input.r - input.g) / delta;
    *h /= 6;
    *h = std::fmod(*h + 1.0f, 1.0f);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RealType simpleToneMap_s(RealType value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    return static_cast<RealType>(1) - std::exp(-value);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RealType sRGB_degamma_s(RealType value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= static_cast<RealType>(0.04045))
        return value / static_cast<RealType>(12.92);
    return std::pow((value + static_cast<RealType>(0.055)) / static_cast<RealType>(1.055),
                    static_cast<RealType>(2.4));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RealType sRGB_gamma_s(RealType value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= static_cast<RealType>(0.0031308))
        return static_cast<RealType>(12.92) * value;
    return static_cast<RealType>(1.055) *
        std::pow(value, static_cast<RealType>(1 / 2.4)) -
        static_cast<RealType>(0.055);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RGBTemplate<RealType> sRGB_degamma(
    const RGBTemplate<RealType> &value) {
    return RGBTemplate<RealType>(
        sRGB_degamma_s(value.r),
        sRGB_degamma_s(value.g),
        sRGB_degamma_s(value.b));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE RGBTemplate<RealType> sRGB_gamma(
    const RGBTemplate<RealType> &value) {
    return RGBTemplate<RealType>(
        sRGB_gamma_s(value.r),
        sRGB_gamma_s(value.g),
        sRGB_gamma_s(value.b));
}



extern const float xbar_CIE1931_deg2[];
extern const float ybar_CIE1931_deg2[];
extern const float zbar_CIE1931_deg2[];

void computeDiscretizedXyzCmfs(
    uint32_t numBands,
    float* centerWavelengths,
    float* xs, float* ys, float* zs, float* integralCmf);



using Point3D = Point3DTemplate<float>;
using Vector3D = Vector3DTemplate<float>;
using Normal3D = Normal3DTemplate<float>;
using Vector4D = Vector4DTemplate<float>;
using TexCoord2D = TexCoord2DTemplate<float>;
using Matrix3x3 = Matrix3x3Template<float>;
using Matrix4x4 = Matrix4x4Template<float>;
using Quaternion = QuaternionTemplate<float>;
using BoundingBox3D = BoundingBox3DTemplate<float>;
using RGBSpectrum = RGBTemplate<float>;

} // namespace rtc8
