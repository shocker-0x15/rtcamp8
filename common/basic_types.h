﻿#include "common_shared.h"

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
    return isnan(x);
#else
    return std::isnan(x);
#endif
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE bool isinf(RealType x) {
#if defined(__CUDA_ARCH__)
    return isinf(x);
#else
    return std::isinf(x);
#endif
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE bool isfinite(RealType x) {
#if defined(__CUDA_ARCH__)
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE void sincos(RealType angle, RealType* s, RealType* c) {
#if defined(__CUDA_ARCH__)
    sincos(x, s, c);
#else
    *s = std::sin(angle);
    *c = std::cos(angle);
#endif
}



template <typename RealType>
struct Point3DTemplate {
    RealType x, y, z;
    
    CUDA_COMMON_FUNCTION Point3DTemplate() {}
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate(RealType v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION constexpr Point3DTemplate(RealType xx, RealType yy, RealType zz) :
        x(xx), y(yy), z(zz) {}

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

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr Point3DTemplate Zero() {
        return Point3DTemplate(0, 0, 0);
    }
};



template <typename RealType, bool isNormal = false>
struct Vector3DTemplate {
    RealType x, y, z;
    
    CUDA_COMMON_FUNCTION Vector3DTemplate() {}
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate(RealType v) :
        x(v), y(v), z(v) {}
    CUDA_COMMON_FUNCTION constexpr Vector3DTemplate(RealType xx, RealType yy, RealType zz) :
        x(xx), y(yy), z(zz) {}
    CUDA_COMMON_FUNCTION constexpr explicit Vector3DTemplate(const Point3DTemplate<RealType> &v) :
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
    CUDA_COMMON_FUNCTION constexpr void makeCoordinateSystem(Vector3DTemplate* vx, Vector3DTemplate* vy) const {
        RealType sign = z >= 0 ? 1 : -1;
        RealType a = -1 / (sign + z);
        RealType b = x * y * a;
        *vx = Vector3DTemplate(1 + sign * x * x * a, sign * b, -sign * x);
        *vy = Vector3DTemplate(b, sign + y * y * a, -y);
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
using Normal3DTemplate = Vector3DTemplate<RealType, true>;



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

    CUDA_COMMON_FUNCTION constexpr RealType determinant() const {
        return (c0[0] * (c1[1] * c2[2] - c2[1] * c1[2]) -
                c1[0] * (c0[1] * c2[2] - c2[1] * c0[2]) +
                c2[0] * (c0[1] * c1[2] - c1[1] * c0[2]));
    }

    CUDA_COMMON_FUNCTION constexpr Matrix3x3Template& transpose() {
        swap(m10, m01); swap(m20, m02);
        swap(m21, m12);
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
        swap(&m10, &m01); swap(&m20, &m02); swap(&m30, &m03);
        swap(&m21, &m12); swap(&m31, &m13);
        swap(&m32, &m23);
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
        *translation = static_cast<Vector3D>(mat.c3);

        // JP: 拡大縮小成分
        // EN: Scale component
        *scale = Vector3D(
            static_cast<Vector3D>(mat.c0).length(),
            static_cast<Vector3D>(mat.c1).length(),
            static_cast<Vector3D>(mat.c2).length());

        // JP: 上記成分を排除
        // EN: Remove the above components
        mat.c3 = Vector4DTemplate<RealType>(0, 0, 0, 1);
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

    CUDA_COMMON_FUNCTION /*constexpr*/ RealType &squaredLength() const {
        return pow2(x) + pow2(y) + pow2(z) + pow2(w);
    }
    CUDA_COMMON_FUNCTION /*constexpr*/ RealType &length() const {
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

    CUDA_COMMON_FUNCTION CUDA_INLINE static constexpr QuaternionTemplate Identity() {
        return QuaternionTemplate(0, 0, 0, 1);
    }
};



// ----------------------------------------------------------------
// Point3D operators and functions

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
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> operator-(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    Vector3DTemplate<RealType> ret(va.x - vb.x, va.y - vb.y, va.z - vb.z);
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
    using rtc8::min;
    return Point3DTemplate<RealType>(min(va.x, vb.x), min(va.y, vb.y), min(va.z, vb.z));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> max(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    using rtc8::max;
    return Point3DTemplate<RealType>(max(va.x, vb.x), max(va.y, vb.y), max(va.z, vb.z));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> squaredDistance(
    const Point3DTemplate<RealType> &va, const Point3DTemplate<RealType> &vb) {
    using rtc8::min;
    Vector3DTemplate<RealType> vector = vb - va;
    return vector.squaredLength();
}

// END: Point3D operators and functions
// ----------------------------------------------------------------



// ----------------------------------------------------------------
// Vector3D/Normal3D operators and functions

#define DEFINE_VECTOR3_OP_EQ(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator==( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        return va.x == vb.x && va.y == vb.y && va.z == vb.z; \
    }
DEFINE_VECTOR3_OP_EQ(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_OP_EQ(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_EQ

#define DEFINE_VECTOR3_OP_NEQ(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!=( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        return va.x != vb.x || va.y != vb.y || va.z != vb.z; \
    }
DEFINE_VECTOR3_OP_NEQ(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_OP_NEQ(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_NEQ

#define DEFINE_VECTOR3_OP_ADD(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> operator+( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        TypeA ## <RealType> ret = va; \
        ret += vb; \
        return ret; \
    }
DEFINE_VECTOR3_OP_ADD(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_OP_ADD(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_ADD

#define DEFINE_VECTOR3_OP_SUBTRACT(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> operator-( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        TypeA ## <RealType> ret = va; \
        ret -= vb; \
        return ret; \
    }
DEFINE_VECTOR3_OP_SUBTRACT(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_OP_SUBTRACT(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_SUBTRACT

#define DEFINE_VECTOR3_OP_MULTIPLY(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> operator*( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        TypeA ## <RealType> ret = va; \
        ret *= vb; \
        return ret; \
    }
DEFINE_VECTOR3_OP_MULTIPLY(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_OP_MULTIPLY(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_MULTIPLY

#define DEFINE_VECTOR3_OP_MULTIPLY_SCALAR(Type) \
    template <typename RealType, typename ScalarType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Type ## <RealType> operator*( \
        const Type ## <RealType> &v, ScalarType s) { \
        Type ## <RealType> ret = v; \
        ret *= s; \
        return ret; \
    }
DEFINE_VECTOR3_OP_MULTIPLY_SCALAR(Vector3DTemplate);
DEFINE_VECTOR3_OP_MULTIPLY_SCALAR(Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_MULTIPLY_SCALAR

#define DEFINE_VECTOR3_OP_MULTIPLY_SCALAR(Type) \
    template <typename ScalarType, typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Type ## <RealType> operator*( \
        ScalarType s, const Type ## <RealType> &v) { \
        Type ## <RealType> ret = v; \
        ret *= s; \
        return ret; \
    }
DEFINE_VECTOR3_OP_MULTIPLY_SCALAR(Vector3DTemplate);
DEFINE_VECTOR3_OP_MULTIPLY_SCALAR(Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_MULTIPLY_SCALAR

#define DEFINE_VECTOR3_OP_DIVIDE(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> operator/( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        TypeA ## <RealType> ret = va; \
        ret /= vb; \
        return ret; \
    }
DEFINE_VECTOR3_OP_DIVIDE(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_OP_DIVIDE(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_DIVIDE

#define DEFINE_VECTOR3_OP_DIVIDE_SCALAR(Type) \
    template <typename RealType, typename ScalarType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Type ## <RealType> operator/( \
        const Type ## <RealType> &v, ScalarType s) { \
        Type ## <RealType> ret = v; \
        ret /= s; \
        return ret; \
    }
DEFINE_VECTOR3_OP_DIVIDE_SCALAR(Vector3DTemplate);
DEFINE_VECTOR3_OP_DIVIDE_SCALAR(Normal3DTemplate);
#undef DEFINE_VECTOR3_OP_DIVIDE_SCALAR

#define DEFINE_VECTOR3_NORMALIZE(Type) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Type ## <RealType> normalize( \
        const Type ## <RealType> &v) { \
        RealType l = v.length(); \
        return v / l; \
    }
DEFINE_VECTOR3_NORMALIZE(Vector3DTemplate);
DEFINE_VECTOR3_NORMALIZE(Normal3DTemplate);
#undef DEFINE_VECTOR3_NORMALIZE

#define DEFINE_VECTOR3_DOT(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr RealType dot( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        return va.x * vb.x + va.y * vb.y + va.z * vb.z; \
    }
DEFINE_VECTOR3_DOT(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_DOT(Vector3DTemplate, Normal3DTemplate);
DEFINE_VECTOR3_DOT(Normal3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_DOT(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_DOT

#define DEFINE_VECTOR3_ABSDOT(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ RealType absDot( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        return std::fabs(dot(va, vb)); \
    }
DEFINE_VECTOR3_ABSDOT(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_ABSDOT(Vector3DTemplate, Normal3DTemplate);
DEFINE_VECTOR3_ABSDOT(Normal3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_ABSDOT(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_ABSDOT

#define DEFINE_VECTOR3_CROSS(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> cross( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        return TypeA ## <RealType>( \
            va.y * vb.z - va.z * vb.y, \
            va.z * vb.x - va.x * vb.z, \
            va.x * vb.y - va.y * vb.x); \
    }
DEFINE_VECTOR3_CROSS(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_CROSS(Vector3DTemplate, Normal3DTemplate);
DEFINE_VECTOR3_CROSS(Normal3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_CROSS(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_CROSS

#define DEFINE_VECTOR3_MIN(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> min( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        using rtc8::min; \
        return TypeA ## <RealType>(min(va.x, vb.x), min(va.y, vb.y), min(va.z, vb.z)); \
    }
DEFINE_VECTOR3_MIN(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_MIN(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_MIN

#define DEFINE_VECTOR3_MAX(TypeA, TypeB) \
    template <typename RealType> \
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr TypeA ## <RealType> max( \
        const TypeA ## <RealType> &va, const TypeB ## <RealType> &vb) { \
        using rtc8::max; \
        return TypeA ## <RealType>(max(va.x, vb.x), max(va.y, vb.y), max(va.z, vb.z)); \
    }
DEFINE_VECTOR3_MAX(Vector3DTemplate, Vector3DTemplate);
DEFINE_VECTOR3_MAX(Normal3DTemplate, Normal3DTemplate);
#undef DEFINE_VECTOR3_MAX

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ Vector3DTemplate<RealType> halfVector(
    const Vector3DTemplate<RealType> &va, const Vector3DTemplate<RealType> &vb) {
    return normalize(va + vb);
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
    using rtc8::min;
    return Vector4DTemplate<RealType>(min(va.x, vb.x), min(va.y, vb.y), min(va.z, vb.z), min(va.w, vb.w));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector4DTemplate<RealType> max(
    const Vector4DTemplate<RealType> &va, const Vector4DTemplate<RealType> &vb) {
    using rtc8::max;
    return Vector4DTemplate<RealType>(max(va.x, vb.x), max(va.y, vb.y), max(va.z, vb.z), max(va.w, vb.w));
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
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Vector3DTemplate<RealType> operator*(
    const Matrix3x3Template<RealType> &mat, const Vector3DTemplate<RealType> &v) {
    return Vector3DTemplate<RealType>(dot(mat.row(0), v), dot(mat.row(1), v), dot(mat.row(2), v));
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Point3DTemplate<RealType> operator*(
    const Matrix3x3Template<RealType> &mat, const Point3DTemplate<RealType> &p) {
    Vector3DTemplate<RealType> v(p);
    return Point3DTemplate<RealType>(dot(mat.row(0), v), dot(mat.row(1), v), dot(mat.row(2), v));
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
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr Matrix4x4Template<RealType> invert(
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
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType>::QuaternionTemplate<RealType>(
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
    return qRotate(angle, Vector3DTemplate<RealType>::Ex);
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotateY(
    RealType angle) {
    return qRotate(angle, Vector3DTemplate<RealType>::Ey);
}
template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> qRotateZ(
    RealType angle) {
    return qRotate(angle, Vector3DTemplate<RealType>::Ez);
}

template <typename RealType>
CUDA_COMMON_FUNCTION CUDA_INLINE /*constexpr*/ QuaternionTemplate<RealType> slerp(
    RealType t, const QuaternionTemplate<RealType> &qa, const QuaternionTemplate<RealType> &qb) {
    RealType cosTheta = dot(qa, qb);
    if (cosTheta > static_cast<RealType>(0.9995))
        return normalize((1 - t) * qa + t * qb);
    else {
        RealType theta = std::acos(clamp(cosTheta, static_cast<RealType>(-1), static_cast<RealType>(1)));
        RealType thetap = theta * t;
        QuaternionTemplate<RealType> qPerp = normalize(qb - qa * cosTheta);
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
            RealType n = fabs(curR[0][i] - nextR[0][i]) + abs(curR[1][i] - nextR[1][i]) + abs(curR[2][i] - nextR[2][i]);
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
        return static_cast<PointType>((p - minP) / (maxP - minP));
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

}