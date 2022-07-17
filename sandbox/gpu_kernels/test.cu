#include "../../common/basic_types.h"

using Point3D = rtc8::Point3DTemplate<float>;
using Vector3D = rtc8::Vector3DTemplate<float>;
using Normal3D = rtc8::Normal3DTemplate<float>;
using Vector4D = rtc8::Vector4DTemplate<float>;
using TexCoord2D = rtc8::TexCoord2DTemplate<float>;
using BoundingBox3D = rtc8::BoundingBox3DTemplate<float>;

CUDA_DEVICE_FUNCTION constexpr Point3D func(const Point3D &a, const Point3D &b) {
    Point3D temp = ((5 * -(a + b)) * 2) / 3.0f;
    temp += a;
    temp *= temp[0];
    //temp /= temp[1];
    //temp[2] += 5;
    Vector3D temp1(temp);
    temp = Point3D(temp1 * temp1.squaredLength());
    return temp;
}

CUDA_DEVICE_KERNEL void kernel0(Point3D* input, Point3D* output) {
    constexpr Point3D c = func(Point3D(1, 2, 3), Point3D(4, 5, 6));
    Point3D d = c;
    d /= d[1];
    d[2] += 5;
    *output = *input + c + d;
}
