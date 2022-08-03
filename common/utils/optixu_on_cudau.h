#pragma once

#include "cuda_util.h"
#include "optix_util.h"

#if defined(OPTIXU_Platform_CodeCompletion)
enum cudaSurfaceBoundaryMode {
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp,
    cudaBoundaryModeTrap,
};
#endif

namespace optixu {
    template <typename T, typename... Ts>
    inline constexpr bool is_any_v = std::disjunction_v<std::is_same<T, Ts>...>;



    template <typename T>
    class NativeBlockBuffer2D {
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        static constexpr size_t roundUp(size_t x) {
            x--;
            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;
            x |= x >> 32;
            x++;
            return x;
        }
        static constexpr bool isPowOf2(size_t x) {
            return (x & (x - 1)) == 0;
        }
        template <typename TargetType>
        RT_DEVICE_FUNCTION static constexpr bool isNativeType() {
            return is_any_v<T,
                float, float2/*, float3*/, float4,
                int32_t, int2/*, int3*/, int4,
                uint32_t, uint2/*, uint3*/, uint4>; // support more types?
        }

        static constexpr size_t s_stride = roundUp(sizeof(T));
        static_assert(s_stride % sizeof(uint32_t) == 0 && s_stride >= 4 && s_stride <= 16,
                      "Unsupported size of type.");

        template <typename TargetType>
        union Alias {
            using ProxyTypes = std::tuple<uint32_t, uint2, uint4, uint4>;
            static constexpr uint32_t s_proxyIndex = sizeof(TargetType) / 4 - 1;
            using ProxyType = std::tuple_element_t<s_proxyIndex, ProxyTypes>;

            TargetType asTargetType;
            ProxyType asProxyType;
            RT_DEVICE_FUNCTION Alias() {}
        };
#endif
        CUsurfObject m_surfObject;

    public:
        RT_COMMON_FUNCTION NativeBlockBuffer2D() : m_surfObject(0) {}
        RT_COMMON_FUNCTION NativeBlockBuffer2D(CUsurfObject surfObject) : m_surfObject(surfObject) {};
        RT_COMMON_FUNCTION operator CUsurfObject() const {
            return m_surfObject;
        }

        RT_COMMON_FUNCTION NativeBlockBuffer2D &operator=(CUsurfObject surfObject) {
            m_surfObject = surfObject;
            return *this;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION T read(
            uint2 idx, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const {
            if constexpr (isNativeType<T>()) {
                return surf2Dread<T>(
                    m_surfObject, idx.x * s_stride, idx.y,
                    boundaryMode);
            }
            else {
                Alias<T> alias;
                alias.asProxyType = surf2Dread<typename Alias<T>::ProxyType>(
                    m_surfObject, idx.x * s_stride, idx.y,
                    boundaryMode);
                return alias.asTargetType;
            }
            return T();
        }
        RT_DEVICE_FUNCTION void write(
            uint2 idx, const T &value, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const {
            if constexpr (isNativeType<T>()) {
                surf2Dwrite(
                    value, m_surfObject, idx.x * s_stride, idx.y,
                    boundaryMode);
            }
            else {
                Alias<T> alias;
                alias.asTargetType = value;
                surf2Dwrite<typename Alias<T>::ProxyType>(
                    alias.asProxyType, m_surfObject, idx.x * s_stride, idx.y,
                    boundaryMode);
            }
        }
        template <size_t offsetInBytes, typename U>
        RT_DEVICE_FUNCTION void writeWithOffset(
            uint2 idx, const U &value, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const {
            if constexpr (isNativeType<U>()) {
                surf2Dwrite(
                    value, m_surfObject, idx.x * s_stride + offsetInBytes, idx.y,
                    boundaryMode);
            }
            else {
                Alias<U> alias;
                alias.asTargetType = value;
                surf2Dwrite<typename Alias<U>::ProxyType>(
                    alias.asProxyType, m_surfObject, idx.x * s_stride + offsetInBytes, idx.y,
                    boundaryMode);
            }
        }

        RT_DEVICE_FUNCTION T read(
            int2 idx, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const {
            return read(make_uint2(idx.x, idx.y), boundaryMode);
        }
        RT_DEVICE_FUNCTION void write(
            int2 idx, const T &value, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const {
            write(make_uint2(idx.x, idx.y), value, boundaryMode);
        }
        template <size_t offsetInBytes, typename U>
        RT_DEVICE_FUNCTION void writeWithOffset(
            int2 idx, const U &value, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const {
            writeWithType<offsetInBytes>(make_uint2(idx.x, idx.y), value, boundaryMode);
        }
#endif
    };



    template <typename T, uint32_t log2BlockWidth>
    class BlockBuffer2D {
        T* m_rawBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numXBlocks;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION constexpr uint32_t calcLinearIndex(uint32_t idxX, uint32_t idxY) const {
            if constexpr (log2BlockWidth > 0) {
                constexpr uint32_t blockWidth = 1 << log2BlockWidth;
                constexpr uint32_t mask = blockWidth - 1;
                uint32_t blockIdxX = idxX >> log2BlockWidth;
                uint32_t blockIdxY = idxY >> log2BlockWidth;
                uint32_t blockOffset = (blockIdxY * m_numXBlocks + blockIdxX) * (blockWidth * blockWidth);
                uint32_t idxXInBlock = idxX & mask;
                uint32_t idxYInBlock = idxY & mask;
                uint32_t linearIndexInBlock = idxYInBlock * blockWidth + idxXInBlock;
                return blockOffset + linearIndexInBlock;
            }
            else {
                return m_width * idxY + idxX;
            }
            return 0;
        }
#endif

    public:
        RT_COMMON_FUNCTION BlockBuffer2D() {}
        RT_COMMON_FUNCTION BlockBuffer2D(T* rawBuffer, uint32_t width, uint32_t height) :
            m_rawBuffer(rawBuffer), m_width(width), m_height(height) {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            m_numXBlocks = ((width + mask) & ~mask) >> log2BlockWidth;
        }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        RT_DEVICE_FUNCTION uint2 getSize() const {
            return make_uint2(m_width, m_height);
        }

        RT_DEVICE_FUNCTION const T &operator[](uint2 idx) const {
            optixuAssert(idx.x < m_width && idx.y < m_height,
                         "Out of bounds: %u, %u", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
        RT_DEVICE_FUNCTION T &operator[](uint2 idx) {
            optixuAssert(idx.x < m_width && idx.y < m_height,
                         "Out of bounds: %u, %u", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
        RT_DEVICE_FUNCTION const T &operator[](int2 idx) const {
            optixuAssert(idx.x >= 0 && idx.x < m_width && idx.y >= 0 && idx.y < m_height,
                         "Out of bounds: %d, %d", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }
        RT_DEVICE_FUNCTION T &operator[](int2 idx) {
            optixuAssert(idx.x >= 0 && idx.x < m_width && idx.y >= 0 && idx.y < m_height,
                         "Out of bounds: %d, %d", idx.x, idx.y);
            return m_rawBuffer[calcLinearIndex(idx.x, idx.y)];
        }

        RT_DEVICE_FUNCTION T read(uint2 idx) const {
            return (*this)[idx];
        }
        RT_DEVICE_FUNCTION void write(uint2 idx, const T &value) {
            (*this)[idx] = value;
        }

        RT_DEVICE_FUNCTION T read(int2 idx) const {
            return (*this)[idx];
        }
        RT_DEVICE_FUNCTION void write(int2 idx, const T &value) {
            (*this)[idx] = value;
        }
#endif
    };

#if !defined(__CUDA_ARCH__)
    template <typename T, uint32_t log2BlockWidth>
    class HostBlockBuffer2D {
        cudau::TypedBuffer<T> m_rawBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numXBlocks;
        T* m_mappedPointer;

        constexpr uint32_t calcLinearIndex(uint32_t x, uint32_t y) const {
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t blockIdxX = x >> log2BlockWidth;
            uint32_t blockIdxY = y >> log2BlockWidth;
            uint32_t blockOffset = (blockIdxY * m_numXBlocks + blockIdxX) * (blockWidth * blockWidth);
            uint32_t idxXInBlock = x & mask;
            uint32_t idxYInBlock = y & mask;
            uint32_t linearIndexInBlock = idxYInBlock * blockWidth + idxXInBlock;
            return blockOffset + linearIndexInBlock;
        }

    public:
        HostBlockBuffer2D() : m_mappedPointer(nullptr) {}
        HostBlockBuffer2D(HostBlockBuffer2D &&b) {
            m_width = b.m_width;
            m_height = b.m_height;
            m_numXBlocks = b.m_numXBlocks;
            m_mappedPointer = b.m_mappedPointer;
            m_rawBuffer = std::move(b);
        }
        HostBlockBuffer2D &operator=(HostBlockBuffer2D &&b) {
            m_rawBuffer.finalize();

            m_width = b.m_width;
            m_height = b.m_height;
            m_numXBlocks = b.m_numXBlocks;
            m_mappedPointer = b.m_mappedPointer;
            m_rawBuffer = std::move(b.m_rawBuffer);

            return *this;
        }

        void initialize(CUcontext context, cudau::BufferType type, uint32_t width, uint32_t height) {
            m_width = width;
            m_height = height;
            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            m_numXBlocks = ((width + mask) & ~mask) >> log2BlockWidth;
            uint32_t numYBlocks = ((height + mask) & ~mask) >> log2BlockWidth;
            uint32_t numElements = numYBlocks * m_numXBlocks * blockWidth * blockWidth;
            m_rawBuffer.initialize(context, type, numElements);
        }
        void finalize() {
            m_rawBuffer.finalize();
        }

        void resize(uint32_t width, uint32_t height) {
            if (!m_rawBuffer.isInitialized())
                throw std::runtime_error("Buffer is not initialized.");

            if (m_width == width && m_height == height)
                return;

            HostBlockBuffer2D newBuffer;
            newBuffer.initialize(m_rawBuffer.getCUcontext(), m_rawBuffer.getBufferType(), width, height);

            constexpr uint32_t blockWidth = 1 << log2BlockWidth;
            constexpr uint32_t mask = blockWidth - 1;
            uint32_t numSrcYBlocks = ((m_height + mask) & ~mask) >> log2BlockWidth;
            uint32_t numDstYBlocks = ((height + mask) & ~mask) >> log2BlockWidth;
            uint32_t numXBlocksToCopy = std::min(m_numXBlocks, newBuffer.m_numXBlocks);
            uint32_t numYBlocksToCopy = std::min(numSrcYBlocks, numDstYBlocks);
            if (numXBlocksToCopy == m_numXBlocks) {
                size_t numBytesToCopy = (numXBlocksToCopy * numYBlocksToCopy * blockWidth * blockWidth) * sizeof(T);
                CUDADRV_CHECK(cuMemcpyDtoD(newBuffer.m_rawBuffer.getCUdeviceptr(),
                                           m_rawBuffer.getCUdeviceptr(),
                                           numBytesToCopy));
            }
            else {
                for (uint32_t yb = 0; yb < numYBlocksToCopy; ++yb) {
                    size_t srcOffset = (m_numXBlocks * blockWidth * blockWidth * yb) * sizeof(T);
                    size_t dstOffset = (newBuffer.m_numXBlocks * blockWidth * blockWidth * yb) * sizeof(T);
                    size_t numBytesToCopy = (numXBlocksToCopy * blockWidth * blockWidth) * sizeof(T);
                    CUDADRV_CHECK(cuMemcpyDtoD(newBuffer.m_rawBuffer.getCUdeviceptr() + dstOffset,
                                               m_rawBuffer.getCUdeviceptr() + srcOffset,
                                               numBytesToCopy));
                }
            }

            *this = std::move(newBuffer);
        }

        CUcontext getCUcontext() const {
            return m_rawBuffer.getCUcontext();
        }
        cudau::BufferType getBufferType() const {
            return m_rawBuffer.getBufferType();
        }

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        CUdeviceptr getCUdeviceptr() const {
            return m_rawBuffer.getCUdeviceptr();
        }
        bool isInitialized() const {
            return m_rawBuffer.isInitialized();
        }

        void map() {
            m_mappedPointer = reinterpret_cast<T*>(m_rawBuffer.map());
        }
        void unmap() {
            m_rawBuffer.unmap();
            m_mappedPointer = nullptr;
        }
        const T &operator()(uint32_t x, uint32_t y) const {
            return m_mappedPointer[calcLinearIndex(x, y)];
        }
        T &operator()(uint32_t x, uint32_t y) {
            return m_mappedPointer[calcLinearIndex(x, y)];
        }

        BlockBuffer2D<T, log2BlockWidth> getBlockBuffer2D() const {
            return BlockBuffer2D<T, log2BlockWidth>(m_rawBuffer.getDevicePointer(), m_width, m_height);
        }
    };
#endif // !defined(__CUDA_ARCH__)
}

#if !defined(__CUDA_ARCH__)

template <>
cudau::Buffer::operator optixu::BufferView() const {
    return optixu::BufferView(getCUdeviceptr(), numElements(), stride());
}

//inline optixu::BufferView getView(const cudau::Buffer &buffer) {
//    return optixu::BufferView(buffer.getCUdeviceptr(), buffer.numElements(), buffer.stride());
//}
#endif // !defined(__CUDA_ARCH__)
