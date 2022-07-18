#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "../common/basic_types.h"

using Point3D = rtc8::Point3DTemplate<float>;
using Vector3D = rtc8::Vector3DTemplate<float>;
using Normal3D = rtc8::Normal3DTemplate<float>;
using Vector4D = rtc8::Vector4DTemplate<float>;
using TexCoord2D = rtc8::TexCoord2DTemplate<float>;
using BoundingBox3D = rtc8::BoundingBox3DTemplate<float>;
using Matrix3x3 = rtc8::Matrix3x3Template<float>;
using Matrix4x4 = rtc8::Matrix4x4Template<float>;
using Quaternion = rtc8::QuaternionTemplate<float>;

constexpr Point3D func(const Point3D &a, const Point3D &b) {
    Point3D temp = ((5 * -(a + b)) * 2) / 3.0f;
    temp += a;
    temp *= temp[0];
    //temp /= temp[1];
    //temp[2] += 5;
    Vector3D temp1(temp);
    temp = Point3D(temp1 * temp1.squaredLength());
    return temp;
}

void kernel0(Point3D* input, Point3D* output) {
    constexpr Point3D c = func(Point3D(1, 2, 3), Point3D(4, 5, 6));
    Point3D d = c;
    d /= d[1];
    d[2] += 5;
    *output = *input + c + d;
}

int32_t main(int32_t argc, const char* argv[]) {
    float theta = rtc8::pi_v<float> / 3;
    float phi = rtc8::pi_v<float> / 4;
    Vector3D dir = Vector3D::fromPolarYUp(phi, theta);
    dir.toPolarYUp(&theta, &phi);

    constexpr Vector3D va(5, 2, 8);
    constexpr Vector3D vb(2, 4, 6);
    constexpr Vector3D vc = min(va, vb);
    constexpr Normal3D nc = vc;

    constexpr Vector4D v4(vc, 5.0f);

    constexpr BoundingBox3D bboxA(static_cast<Point3D>(va), static_cast<Point3D>(vb));
    constexpr Point3D centroid = bboxA.calcCentroid();

    constexpr BoundingBox3D bboxB(Point3D(-2, -3, -1), Point3D(4, 6, 2));
    constexpr float saB = bboxB.calcHalfSurfaceArea();
    constexpr float volB = bboxB.calcVolume();
    constexpr Point3D localP = bboxB.calcLocalCoordinates(Point3D(1.5f, 0.0f, 1.75f));
    constexpr uint32_t widestDim = bboxB.calcWidestDimension();
    constexpr BoundingBox3D bboxC = unify(bboxA, bboxB);

    constexpr Matrix3x3 matIdentity = Matrix3x3::Identity();
    constexpr Vector3D trVa = matIdentity * va;
    constexpr Point3D trCentroid = matIdentity * centroid;
    constexpr Matrix3x3 matScale = rtc8::scale3x3<float>(1, 2, 3);

    constexpr Matrix4x4 mat4x4A = rtc8::translate4x4(Vector3D(-2, 3, 7));
    //constexpr float tempA[] = {
    //    1, 0, 0, 0,
    //    0, 1, 0, 0,
    //    0, 0, 1, 0,
    //    -2, 3, 7, 1
    //};
    //constexpr Matrix4x4 mat4x4A(tempA);
    constexpr BoundingBox3D trBboxC = mat4x4A * bboxC;
    Matrix4x4 invMat4x4A = invert(mat4x4A);

    Matrix4x4 tempA = rtc8::rotate4x4(rtc8::pi_v<float> / 3, Vector3D(-4, -1.5f, 3.0f).normalize()) *
        rtc8::translate4x4(-3.2f, -2.7f, 5.0f);
    auto tempB = invert(tempA) * tempA;

    return 0;
}
