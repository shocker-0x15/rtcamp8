#pragma once

#include "basic_types.h"

// JP: Callable Programや関数ポインターによる動的な関数呼び出しを
//     無くした場合の性能を見たい場合にこのマクロを有効化する。
// EN: Enable this switch when you want to see performance
//     without dynamic function calls by callable programs or function pointers.
static constexpr bool useGenericBSDF = false;
//#define HARD_CODED_BSDF DichromaticBRDF
#define HARD_CODED_BSDF SimplePBR_BRDF
//#define HARD_CODED_BSDF LambertBRDF

#define PROCESS_DYNAMIC_FUNCTIONS \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromNormalMap2ch), \
    PROCESS_DYNAMIC_FUNCTION(readModifiedNormalFromHeightMap), \
    PROCESS_DYNAMIC_FUNCTION(setupLambertBRDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_sampleF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(LambertBRDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(setupDichromaticBRDF), \
    PROCESS_DYNAMIC_FUNCTION(setupSimplePBR_BRDF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_getSurfaceParameters), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_sampleF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluateF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluatePDF), \
    PROCESS_DYNAMIC_FUNCTION(DichromaticBRDF_evaluateDHReflectanceEstimate), \
    PROCESS_DYNAMIC_FUNCTION(ImageBasedEnvironmentalLight_sample),

enum CallableProgram {
#define PROCESS_DYNAMIC_FUNCTION(Func) CallableProgram_ ## Func
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
    NumCallablePrograms
};

// JP: OptiXに関してホストコードが必要とするCallable Programsのエントリーポイント名。
// EN: Entry point names of callable programs required by the host code regarding OptiX.
constexpr const char* callableProgramEntryPoints[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) RT_DC_NAME_STR(#Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};

// JP: Pure CUDAに関してホストコードが必要とする関数ポインターのシンボル名。
// EN: Symbol names of function pointers required by the host code regarding pure CUDA.
#define CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(name) "ptr_" #name
constexpr const char* callableProgramPointerNames[] = {
#define PROCESS_DYNAMIC_FUNCTION(Func) CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR(Func)
    PROCESS_DYNAMIC_FUNCTIONS
#undef PROCESS_DYNAMIC_FUNCTION
};
#undef CUDA_CALLABLE_PROGRAM_POINTER_NAME_STR

#undef PROCESS_DYNAMIC_FUNCTIONS

// JP: Callable Programのインデックスから対応する関数ポインターへのマップ。
// EN: Map from callable programs to corresponding function pointers.
#if (defined(__CUDA_ARCH__) && defined(PURE_CUDA)) || defined(OPTIXU_Platform_CodeCompletion)
CUDA_CONSTANT_MEM void* c_callableToPointerMap[NumCallablePrograms];
#endif

#if defined(PURE_CUDA)
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name) \
        extern "C" CUDA_DEVICE_MEM auto ptr_ ## name = RT_DC_NAME(name)
#else
#   define CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(name)
#endif



namespace rtc8 {
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



namespace rtc8::shared {

CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint32_t mapPrimarySampleToDiscrete(
    float u01, uint32_t numValues, float* uRemapped = nullptr) {
    uint32_t idx = min(static_cast<uint32_t>(u01 * numValues), numValues - 1);
    if (uRemapped)
        *uRemapped = u01 * numValues - idx;
    return idx;
}



class PCG32RNG {
    uint64_t state;

public:
    CUDA_COMMON_FUNCTION PCG32RNG() {}

    CUDA_COMMON_FUNCTION constexpr void setState(uint64_t _state) {
        state = _state;
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ uint32_t operator()() {
        uint64_t oldstate = state;
        // Advance internal state
        state = oldstate * 6364136223846793005ULL + 1;
        // Calculate output function (XSH RR), uses old state for max ILP
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
    }

    CUDA_COMMON_FUNCTION /*constexpr*/ float getFloat0cTo1o() {
        uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
        return *reinterpret_cast<float*>(&fractionBits) - 1.0f;
    }
};



template <typename RealType>
struct AliasTableEntry {
    uint32_t secondIndex;
    RealType probToPickFirst;

    CUDA_COMMON_FUNCTION AliasTableEntry() {}
    CUDA_COMMON_FUNCTION constexpr AliasTableEntry(uint32_t _secondIndex, RealType _probToPickFirst) :
        secondIndex(_secondIndex), probToPickFirst(_probToPickFirst) {}
};

template <typename RealType>
struct AliasValueMap {
    RealType scaleForFirst;
    RealType scaleForSecond;
    RealType offsetForSecond;
};



template <typename RealType, bool useAliasTable>
class DiscreteDistribution1DTemplate;

template <typename RealType>
class DiscreteDistribution1DTemplate<RealType, false> {
    RealType* m_weights;
    RealType* m_CDF;
    RealType m_integral;
    uint32_t m_numValues;

public:
    DiscreteDistribution1DTemplate(
        RealType* weights, RealType* CDF, RealType integral, uint32_t numValues) :
        m_weights(weights), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped = nullptr) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        u *= m_integral;
        int idx = 0;
        for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= m_numValues)
                continue;
            if (m_CDF[idx + d] <= u)
                idx += d;
        }
        Assert(idx < m_numValues, "Invalid Index!: %u >= %u, u: %g, integ: %g",
               idx, m_numValues, u, m_integral);
        if (remapped) {
            RealType lCDF = m_CDF[idx];
            RealType rCDF = m_integral;
            if (idx < m_numValues - 1)
                rCDF = m_CDF[idx + 1];
            *remapped = (u - lCDF) / (rCDF - lCDF);
            Assert(isfinite(*remapped), "Remapped value is not a finite value %g.",
                   *remapped);
    }
        *prob = m_weights[idx] / m_integral;
        return idx;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
        if (!m_weights || m_integral == 0.0f)
            return 0.0f;
        Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
        return m_weights[idx] / m_integral;
    }

    CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

    CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }

    CUDA_COMMON_FUNCTION uint32_t setNumValues(uint32_t numValues) {
        m_numValues = numValues;
    }

    CUDA_COMMON_FUNCTION const RealType* weights() const {
        return m_weights;
    }

    CUDA_COMMON_FUNCTION const RealType* cdfs() const {
        return m_CDF;
    }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION uint32_t setWeightAt(uint32_t index, RealType value) {
        m_weights[index] = value;
    }

    CUDA_DEVICE_FUNCTION void finalize() {
        uint32_t lastIndex = m_numValues - 1;
        m_integral = m_CDF[lastIndex] + m_weights[lastIndex];
        //printf("%g\n", m_integral);
    }
#endif
};

template <typename RealType>
class DiscreteDistribution1DTemplate<RealType, true> {
    RealType* m_weights;
    const AliasTableEntry<RealType>* m_aliasTable;
    const AliasValueMap<RealType>* m_valueMaps;
    RealType m_integral;
    uint32_t m_numValues;

public:
    DiscreteDistribution1DTemplate(
        RealType* weights, AliasTableEntry<RealType>* aliasTable, AliasValueMap<RealType>* valueMaps,
        RealType integral, uint32_t numValues) :
        m_weights(weights), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
        m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION DiscreteDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped = nullptr) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
        const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
        const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
        if (u < entry.probToPickFirst) {
            if (remapped)
                *remapped = valueMap.scaleForFirst * u;
        }
        else {
            idx = entry.secondIndex;
            if (remapped)
                *remapped = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
        }
        *prob = m_weights[idx] / m_integral;
        return idx;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePMF(uint32_t idx) const {
        if (!m_weights || m_integral == 0.0f)
            return 0.0f;
        Assert(idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
        return m_weights[idx] / m_integral;
    }

    CUDA_COMMON_FUNCTION RealType integral() const { return m_integral; }

    CUDA_COMMON_FUNCTION uint32_t numValues() const { return m_numValues; }

    CUDA_COMMON_FUNCTION uint32_t setNumValues(uint32_t numValues) {
        m_numValues = numValues;
    }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION uint32_t setWeightAt(uint32_t index, RealType value) {
        m_weights[index] = value;
    }
#endif
};



template <typename RealType, bool useAliasTable>
class RegularConstantContinuousDistribution1DTemplate;

template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate<RealType, false> {
    const RealType* m_PDF;
    const RealType* m_CDF;
    RealType m_integral;
    uint32_t m_numValues;

public:
    RegularConstantContinuousDistribution1DTemplate(
        const RealType* PDF, const RealType* CDF, RealType integral, uint32_t numValues) :
        m_PDF(PDF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        int idx = 0;
        for (int d = nextPowerOf2(m_numValues) >> 1; d >= 1; d >>= 1) {
            if (idx + d >= m_numValues)
                continue;
            if (m_CDF[idx + d] <= u)
                idx += d;
        }
        Assert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
        RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
        *probDensity = m_PDF[idx];
        return (idx + t) / m_numValues;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType smp) const {
        Assert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
        int32_t idx = min(m_numValues - 1, static_cast<uint32_t>(smp * m_numValues));
        return m_PDF[idx];
    }

    CUDA_COMMON_FUNCTION RealType integral() const {
        return m_integral;
    }

    CUDA_COMMON_FUNCTION uint32_t numValues() const {
        return m_numValues;
    }
};

template <typename RealType>
class RegularConstantContinuousDistribution1DTemplate<RealType, true> {
    const RealType* m_PDF;
    const AliasTableEntry<RealType>* m_aliasTable;
    const AliasValueMap<RealType>* m_valueMaps;
    RealType m_integral;
    uint32_t m_numValues;

public:
    RegularConstantContinuousDistribution1DTemplate(
        const RealType* PDF, const AliasTableEntry<RealType>* aliasTable, const AliasValueMap<RealType>* valueMaps,
        RealType integral, uint32_t numValues) :
        m_PDF(PDF), m_aliasTable(aliasTable), m_valueMaps(valueMaps),
        m_integral(integral), m_numValues(numValues) {}

    CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}

    CUDA_COMMON_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
        Assert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
        uint32_t idx = mapPrimarySampleToDiscrete(u, m_numValues, &u);
        const AliasTableEntry<RealType> &entry = m_aliasTable[idx];
        const AliasValueMap<RealType> &valueMap = m_valueMaps[idx];
        RealType t;
        if (u < entry.probToPickFirst) {
            t = valueMap.scaleForFirst * u;
        }
        else {
            idx = entry.secondIndex;
            t = valueMap.scaleForSecond * u + valueMap.offsetForSecond;
        }
        *probDensity = m_PDF[idx];
        return (idx + t) / m_numValues;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType smp) const {
        Assert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
        int32_t idx = min(m_numValues - 1, static_cast<uint32_t>(smp * m_numValues));
        return m_PDF[idx];
    }

    CUDA_COMMON_FUNCTION RealType integral() const {
        return m_integral;
    }

    CUDA_COMMON_FUNCTION uint32_t numValues() const {
        return m_numValues;
    }
};



template <typename RealType, bool useAliasTable>
class RegularConstantContinuousDistribution2DTemplate {
    using RegularConstantContinuousDistribution1D =
        RegularConstantContinuousDistribution1DTemplate<RealType, useAliasTable>;

    const RegularConstantContinuousDistribution1D* m_1DDists;
    RegularConstantContinuousDistribution1D m_top1DDist;

public:
    RegularConstantContinuousDistribution2DTemplate(
        const RegularConstantContinuousDistribution1D* _1DDists,
        const RegularConstantContinuousDistribution1D &top1DDist) :
        m_1DDists(_1DDists), m_top1DDist(top1DDist) {}

    CUDA_COMMON_FUNCTION RegularConstantContinuousDistribution2DTemplate() {}

    CUDA_COMMON_FUNCTION void sample(
        RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
        RealType topPDF;
        *d1 = m_top1DDist.sample(u1, &topPDF);
        uint32_t idx1D = mapPrimarySampleToDiscrete(*d1, m_top1DDist.numValues());
        *d0 = m_1DDists[idx1D].sample(u0, probDensity);
        *probDensity *= topPDF;
    }

    CUDA_COMMON_FUNCTION RealType evaluatePDF(RealType d0, RealType d1) const {
        uint32_t idx1D = mapPrimarySampleToDiscrete(d1, m_top1DDist.numValues());
        return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
    }
};



using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float, false>;
using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float, false>;
using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float, false>;
using LightDistribution = DiscreteDistribution1D;



template <typename FuncType>
class DynamicFunction;

template <typename ReturnType, typename... ArgTypes>
class DynamicFunction<ReturnType(ArgTypes...)> {
    using Signature = ReturnType(*)(ArgTypes...);
    optixu::DirectCallableProgramID<ReturnType(ArgTypes...)> m_callableHandle;

public:
    CUDA_COMMON_FUNCTION DynamicFunction() {}
    CUDA_COMMON_FUNCTION constexpr DynamicFunction(uint32_t sbtIndex) : m_callableHandle(sbtIndex) {}

    CUDA_COMMON_FUNCTION constexpr explicit operator uint32_t() const {
        return static_cast<uint32_t>(m_callableHandle);
    }

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION ReturnType operator()(const ArgTypes &... args) const {
#   if defined(PURE_CUDA)
        void* ptr = c_callableToPointerMap[static_cast<uint32_t>(m_callableHandle)];
        auto func = reinterpret_cast<Signature>(ptr);
        return func(args...);
#   else
        return m_callableHandle(args...);
#   endif
    }
#endif
};



struct TexDimInfo {
    uint32_t dimX : 14;
    uint32_t dimY : 14;
    uint32_t isNonPowerOfTwo : 1;
    uint32_t isBCTexture : 1;
    uint32_t isLeftHanded : 1; // for normal map
};



using ReadModifiedNormal = DynamicFunction<
    Normal3D(CUtexObject texture, TexDimInfo dimInfo, TexCoord2D texCoord)>;



struct DirectionType {
    enum InternalEnum : uint32_t {
        IE_LowFreq = 1 << 0,
        IE_HighFreq = 1 << 1,
        IE_Delta0D = 1 << 2,
        IE_Delta1D = 1 << 3,
        IE_NonDelta = IE_LowFreq | IE_HighFreq,
        IE_Delta = IE_Delta0D | IE_Delta1D,
        IE_AllFreq = IE_NonDelta | IE_Delta,

        IE_Reflection = 1 << 4,
        IE_Transmission = 1 << 5,
        IE_Emission = IE_Reflection,
        IE_Acquisition = IE_Reflection,
        IE_WholeSphere = IE_Reflection | IE_Transmission,

        IE_All = IE_AllFreq | IE_WholeSphere,

        IE_Dispersive = 1 << 6,

        IE_LowFreqReflection = IE_LowFreq | IE_Reflection,
        IE_LowFreqTransmission = IE_LowFreq | IE_Transmission,
        IE_LowFreqScattering = IE_LowFreqReflection | IE_LowFreqTransmission,
        IE_HighFreqReflection = IE_HighFreq | IE_Reflection,
        IE_HighFreqTransmission = IE_HighFreq | IE_Transmission,
        IE_HighFreqScattering = IE_HighFreqReflection | IE_HighFreqTransmission,
        IE_Delta0DReflection = IE_Delta0D | IE_Reflection,
        IE_Delta0DTransmission = IE_Delta0D | IE_Transmission,
        IE_Delta0DScattering = IE_Delta0DReflection | IE_Delta0DTransmission,
    };

    InternalEnum value;

    CUDA_DEVICE_FUNCTION DirectionType() { }
    CUDA_DEVICE_FUNCTION constexpr DirectionType(InternalEnum v) : value(v) { }
    CUDA_DEVICE_FUNCTION constexpr DirectionType &operator&=(const DirectionType &r) {
        value = static_cast<InternalEnum>(value & r.value);
        return *this;
    }
    CUDA_DEVICE_FUNCTION constexpr DirectionType &operator|=(const DirectionType &r) {
        value = static_cast<InternalEnum>(value | r.value);
        return *this;
    }
    CUDA_DEVICE_FUNCTION constexpr DirectionType flip() const {
        return static_cast<InternalEnum>(value ^ IE_WholeSphere);
    }

    CUDA_DEVICE_FUNCTION constexpr bool matches(DirectionType t) const {
        uint32_t res = value & t.value;
        return (res & IE_WholeSphere) && (res & IE_AllFreq);
    }
    CUDA_DEVICE_FUNCTION constexpr bool hasNonDelta() const {
        return value & IE_NonDelta;
    }
    CUDA_DEVICE_FUNCTION constexpr bool hasDelta() const {
        return value & IE_Delta;
    }
    CUDA_DEVICE_FUNCTION constexpr bool isDelta() const {
        return (value & IE_Delta) && !(value & IE_NonDelta);
    }
    CUDA_DEVICE_FUNCTION constexpr bool isReflection() const {
        return (value & IE_Reflection) && !(value & IE_Transmission);
    }
    CUDA_DEVICE_FUNCTION constexpr bool isTransmission() const {
        return !(value & IE_Reflection) && (value & IE_Transmission);
    }
    CUDA_DEVICE_FUNCTION constexpr bool isDispersive() const {
        return value & IE_Dispersive;
    }

    CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreq() {
        return IE_LowFreq;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreq() {
        return IE_HighFreq;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0D() {
        return IE_Delta0D;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta1D() {
        return IE_Delta1D;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType NonDelta() {
        return IE_NonDelta;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta() {
        return IE_Delta;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType AllFreq() {
        return IE_AllFreq;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Reflection() {
        return IE_Reflection;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Transmission() {
        return IE_Transmission;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Emission() {
        return IE_Emission;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Acquisition() {
        return IE_Acquisition;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType WholeSphere() {
        return IE_WholeSphere;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType All() {
        return IE_All;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Dispersive() {
        return IE_Dispersive;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreqReflection() {
        return IE_LowFreqReflection;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreqTransmission() {
        return IE_LowFreqTransmission;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType LowFreqScattering() {
        return IE_LowFreqScattering;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreqReflection() {
        return IE_HighFreqReflection;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreqTransmission() {
        return IE_HighFreqTransmission;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType HighFreqScattering() {
        return IE_HighFreqScattering;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0DReflection() {
        return IE_Delta0DReflection;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0DTransmission() {
        return IE_Delta0DTransmission;
    }
    CUDA_DEVICE_FUNCTION static constexpr DirectionType Delta0DScattering() {
        return IE_Delta0DScattering;
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool operator==(
    const DirectionType &l, const DirectionType &r) {
    return l.value == r.value;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr bool operator!=(
    const DirectionType &l, const DirectionType &r) {
    return l.value != r.value;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType operator&(
    const DirectionType &l, const DirectionType &r) {
    DirectionType ret = l;
    ret &= r;
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr DirectionType operator|(
    const DirectionType &l, const DirectionType &r) {
    DirectionType ret = l;
    ret |= r;
    return ret;
}



struct SurfaceMaterial;

struct BSDFBuildFlags {
    enum Value {
        None = 0,
    } value;

    CUDA_COMMON_FUNCTION constexpr BSDFBuildFlags(Value v = None) : value(v) {}

    CUDA_COMMON_FUNCTION operator uint32_t() const {
        return static_cast<uint32_t>(value);
    }
};

struct BSDFQuery {
    Vector3D dirLocal;
    Normal3D geometricNormalLocal;

    CUDA_COMMON_FUNCTION BSDFQuery(
        const Vector3D &dirL, const Normal3D &gNormL) :
        dirLocal(dirL), geometricNormalLocal(gNormL) {}
};

struct BSDFSample {
    float uDir[2];

    CUDA_COMMON_FUNCTION BSDFSample() {}
    CUDA_COMMON_FUNCTION BSDFSample(float uDir0, float uDir1) :
        uDir{ uDir0, uDir1 } {}
};

struct BSDFQueryResult {
    Vector3D dirLocal;
    float dirPDensity;
    DirectionType sampledType;

    CUDA_COMMON_FUNCTION BSDFQueryResult() {}
};

using SetupBSDFBody = DynamicFunction<
    void(const SurfaceMaterial &matData, TexCoord2D texCoord, uint32_t* bodyData, BSDFBuildFlags flags)>;

using BSDFGetSurfaceParameters = DynamicFunction<
    void(const uint32_t* data,
         RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness)>;
using BSDFSampleF = DynamicFunction<
    RGBSpectrum(const uint32_t* data, const BSDFQuery &query, const BSDFSample &sample, BSDFQueryResult* result)>;
using BSDFEvaluateF = DynamicFunction<
    RGBSpectrum(const uint32_t* data, const BSDFQuery &query, const Vector3D &vSampled)>;
using BSDFEvaluatePDF = DynamicFunction<
    float(const uint32_t* data, const BSDFQuery &query, const Vector3D &vSampled)>;
using BSDFEvaluateDHReflectanceEstimate = DynamicFunction<
    RGBSpectrum(const uint32_t* data, const BSDFQuery &query)>;

struct BSDFProcedureSet {
    SetupBSDFBody setupBSDFBody;
    BSDFGetSurfaceParameters getSurfaceParameters;
    BSDFSampleF sampleF;
    BSDFEvaluateF evaluateF;
    BSDFEvaluatePDF evaluatePDF;
    BSDFEvaluateDHReflectanceEstimate evaluateDHReflectanceEstimate;
};



struct LambertianSurfaceMaterial {
    CUtexObject reflectance;
    TexDimInfo reflectanceDimInfo;
};

struct DichromaticSurfaceMaterial {
    CUtexObject diffuse;
    CUtexObject specular;
    CUtexObject smoothness;
    TexDimInfo diffuseDimInfo;
    TexDimInfo specularDimInfo;
    TexDimInfo smoothnessDimInfo;
};

struct SimplePBRSurfaceMaterial {
    CUtexObject baseColor_opacity;
    CUtexObject occlusion_roughness_metallic;
    TexDimInfo baseColor_opacity_dimInfo;
    TexDimInfo occlusion_roughness_metallic_dimInfo;
};



struct SurfaceMaterial {
    uint32_t body[12];
    CUtexObject emittance;
    uint32_t bsdfProcSetSlot;
    SetupBSDFBody setupBSDFBody; // shortcut
};



using EnvLightSample = DynamicFunction<
    RGBSpectrum(const uint32_t* data, float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity)>;



struct ImageBasedEnvironmentalLight {
    CUtexObject texObj;
    RegularConstantContinuousDistribution2D importanceMap;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION RGBSpectrum sample(
        float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) const;
#endif
};




struct EnvironmentalLight {
    uint32_t body[sizeof(ImageBasedEnvironmentalLight) / sizeof(uint32_t)];

    EnvLightSample envLightSample;
    unsigned int enabled : 1;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    CUDA_DEVICE_FUNCTION RGBSpectrum sample(
        float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) const {
        return envLightSample(body, uDir0, uDir1, phi, theta, dirPDensity);
    }
#endif
};



struct PerspectiveCamera {
    float aspect;
    float fovY;
    Point3D position;
    Quaternion orientation;

    CUDA_COMMON_FUNCTION float2 calcScreenPosition(const Point3D &posInWorld) const {
        Matrix3x3 invOri = conjugate(orientation).toMatrix3x3();
        Vector3D posInView = invOri * (posInWorld - position);
        float2 posAtZ1 = make_float2(posInView.x / posInView.z, posInView.y / posInView.z);
        float h = 2 * std::tan(fovY / 2);
        float w = aspect * h;
        return make_float2(1 - (posAtZ1.x + 0.5f * w) / w,
                           1 - (posAtZ1.y + 0.5f * h) / h);
    }
};



struct Vertex {
    Point3D position;
    Normal3D normal;
    Vector3D tangent;
    TexCoord2D texCoord;
};

struct Triangle {
    uint32_t indices[3];
};

struct GeometryInstance {
    const Vertex* vertices;
    const Triangle* triangles;
    CUtexObject normal;
    TexDimInfo normalDimInfo;
    ReadModifiedNormal readModifiedNormal;
    LightDistribution emitterPrimDist;
    uint32_t surfMatSlot;
};

struct StaticTransform {
    Matrix4x4 mat;
    Matrix4x4 invMat;

    CUDA_COMMON_FUNCTION StaticTransform() {}
    CUDA_COMMON_FUNCTION /*constexpr*/ StaticTransform(const Matrix4x4 &_mat) :
        mat(_mat), invMat(invert(_mat)) {}
    CUDA_COMMON_FUNCTION constexpr StaticTransform(
        const Matrix4x4 &_mat, const Matrix4x4 _invMat) :
        mat(_mat), invMat(_invMat) {}

    CUDA_COMMON_FUNCTION constexpr Vector3D operator*(const Vector3D &v) const {
        return mat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4D operator*(const Vector4D &v) const {
        return mat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Point3D operator*(const Point3D &p) const {
        return mat * p;
    }
    CUDA_COMMON_FUNCTION constexpr Normal3D operator*(const Normal3D &n) const {
        // The length of the normal is changed if the transform has scaling, so it requires normalization.
        return Normal3D(invMat.m00 * n.x + invMat.m10 * n.y + invMat.m20 * n.z,
                        invMat.m01 * n.x + invMat.m11 * n.y + invMat.m21 * n.z,
                        invMat.m02 * n.x + invMat.m12 * n.y + invMat.m22 * n.z);
    }
    CUDA_COMMON_FUNCTION constexpr BoundingBox3D operator*(const BoundingBox3D &b) const { return mat * b; }


    CUDA_COMMON_FUNCTION constexpr Vector3D mulInv(const Vector3D& v) const {
        return invMat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Vector4D mulInv(const Vector4D& v) const {
        return invMat * v;
    }
    CUDA_COMMON_FUNCTION constexpr Point3D mulInv(const Point3D& p) const {
        return invMat * p;
    }
    CUDA_COMMON_FUNCTION constexpr Normal3D mulInv(const Normal3D& n) const {
        // The length of the normal is changed if the transform has scaling, so it requires normalization.
        return Normal3D(mat.m00 * n.x + mat.m10 * n.y + mat.m20 * n.z,
                        mat.m01 * n.x + mat.m11 * n.y + mat.m21 * n.z,
                        mat.m02 * n.x + mat.m12 * n.y + mat.m22 * n.z);
    }
    CUDA_COMMON_FUNCTION constexpr BoundingBox3D mulInv(const BoundingBox3D &b) const {
        return invMat * b;
    }
};

struct GeometryGroup {
    const uint32_t* geomInstSlots;
    LightDistribution lightGeomInstDist;
};

struct Instance {
    StaticTransform transform;
    StaticTransform prevTransform;
    float uniformScale;
    uint32_t geomGroupSlot;
};

} // namespace rtc8::shared



#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)

#define print3d(v) v.x, v.y, v.z

namespace rtc8::device {

static constexpr float RayEpsilon = 1e-4;



CUDA_DEVICE_FUNCTION CUDA_INLINE void concentricSampleDisk(float u0, float u1, float* dx, float* dy) {
    float r, theta;
    float sx = 2 * u0 - 1;
    float sy = 2 * u1 - 1;

    if (sx == 0 && sy == 0) {
        *dx = 0;
        *dy = 0;
        return;
    }
    if (sx >= -sy) { // region 1 or 2
        if (sx > sy) { // region 1
            r = sx;
            theta = sy / sx;
        }
        else { // region 2
            r = sy;
            theta = 2 - sx / sy;
        }
    }
    else { // region 3 or 4
        if (sx > sy) {/// region 4
            r = -sy;
            theta = 6 + sx / sy;
        }
        else {// region 3
            r = -sx;
            theta = 4 + sy / sx;
        }
    }
    theta *= pi_v<float> / 4;
    *dx = r * cos(theta);
    *dy = r * sin(theta);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE Vector3D cosineSampleHemisphere(float u0, float u1) {
    float x, y;
    concentricSampleDisk(u0, u1, &x, &y);
    return Vector3D(x, y, std::sqrt(std::fmax(0.0f, 1.0f - x * x - y * y)));
}



// JP: 自己交叉回避のためにレイの原点にオフセットを付加する。
// EN: Add an offset to a ray origin to avoid self-intersection.
CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr Point3D offsetRayOriginNaive(
    const Point3D &p, const Normal3D &geometricNormal) {
    return p + RayEpsilon * geometricNormal;
}

// Reference:
// Chapter 6. A Fast and Robust Method for Avoiding Self-Intersection, Ray Tracing Gems, 2019
CUDA_DEVICE_FUNCTION CUDA_INLINE /*constexpr*/ Point3D offsetRayOrigin(
    const Point3D &p, const Normal3D &geometricNormal) {
    constexpr float kOrigin = 1.0f / 32.0f;
    constexpr float kFloatScale = 1.0f / 65536.0f;
    constexpr float kIntScale = 256.0f;

    int32_t offsetInInt[] = {
        static_cast<int32_t>(kIntScale * geometricNormal.x),
        static_cast<int32_t>(kIntScale * geometricNormal.y),
        static_cast<int32_t>(kIntScale * geometricNormal.z)
    };

    // JP: 数学的な衝突点の座標と、実際の座標の誤差は原点からの距離に比例する。
    //     intとしてオフセットを加えることでスケール非依存に適切なオフセットを加えることができる。
    // EN: The error of the actual coorinates of the intersection point to the mathematical one is proportional to the distance to the origin.
    //     Applying the offset as int makes applying appropriate scale invariant amount of offset possible.
    Point3D newP1(
        __int_as_float(__float_as_int(p.x) + (p.x < 0 ? -1 : 1) * offsetInInt[0]),
        __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -1 : 1) * offsetInInt[1]),
        __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -1 : 1) * offsetInInt[2]));

    // JP: 原点に近い場所では、原点からの距離に依存せず一定の誤差が残るため別処理が必要。
    // EN: A constant amount of error remains near the origin independent of the distance to the origin so we need handle it separately.
    Point3D newP2 = p + kFloatScale * geometricNormal;

    return Point3D(std::fabs(p.x) < kOrigin ? newP2.x : newP1.x,
                   std::fabs(p.y) < kOrigin ? newP2.y : newP1.y,
                   std::fabs(p.z) < kOrigin ? newP2.z : newP1.z);
}



CUDA_DEVICE_FUNCTION CUDA_INLINE constexpr TexCoord2D adjustTexCoord(
    shared::TexDimInfo dimInfo, const TexCoord2D &texCoord) {
    TexCoord2D mTexCoord = texCoord;
    if (dimInfo.isNonPowerOfTwo && dimInfo.isBCTexture) {
        uint32_t bcWidth = (dimInfo.dimX + 3) / 4 * 4;
        uint32_t bcHeight = (dimInfo.dimY + 3) / 4 * 4;
        mTexCoord.u *= static_cast<float>(dimInfo.dimX) / bcWidth;
        mTexCoord.v *= static_cast<float>(dimInfo.dimY) / bcHeight;
    }
    return mTexCoord;
}

template <typename T>
CUDA_DEVICE_FUNCTION CUDA_INLINE T sample(
    CUtexObject texture, shared::TexDimInfo dimInfo, const TexCoord2D &texCoord) {
    TexCoord2D mTexCoord = adjustTexCoord(dimInfo, texCoord);
    return tex2DLod<T>(texture, mTexCoord.u, mTexCoord.v, 0.0f);
}



struct ReferenceFrame {
    Vector3D tangent;
    Vector3D bitangent;
    Normal3D normal;

    CUDA_DEVICE_FUNCTION ReferenceFrame() {}
    CUDA_DEVICE_FUNCTION constexpr ReferenceFrame(
        const Vector3D &_tangent, const Vector3D &_bitangent, const Normal3D &_normal) :
        tangent(_tangent), bitangent(_bitangent), normal(_normal) {}
    CUDA_DEVICE_FUNCTION /*constexpr*/ ReferenceFrame(const Normal3D &_normal) :
        normal(_normal) {
        normal.makeCoordinateSystem(&tangent, &bitangent);
    }
    CUDA_DEVICE_FUNCTION /*constexpr*/ ReferenceFrame(const Normal3D &_normal, const Vector3D &_tangent) :
        tangent(_tangent), normal(_normal) {
        bitangent = cross(normal, tangent);
    }

    CUDA_DEVICE_FUNCTION constexpr Vector3D toLocal(const Vector3D &v) const {
        return Vector3D(dot(tangent, v), dot(bitangent, v), dot(normal, v));
    }
    CUDA_DEVICE_FUNCTION constexpr Vector3D fromLocal(const Vector3D &v) const {
        return Vector3D(dot(Vector3D(tangent.x, bitangent.x, normal.x), v),
                        dot(Vector3D(tangent.y, bitangent.y, normal.y), v),
                        dot(Vector3D(tangent.z, bitangent.z, normal.z), v));
    }
};

CUDA_DEVICE_FUNCTION CUDA_INLINE void applyBumpMapping(
    const Normal3D &modNormalInTF, ReferenceFrame* frameToModify) {
    // JP: 法線から回転軸と回転角(、Quaternion)を求めて対応する接平面ベクトルを求める。
    // EN: calculate a rotating axis and an angle (and quaternion) from the normal then calculate corresponding tangential vectors.
    float projLength = std::sqrt(modNormalInTF.x * modNormalInTF.x + modNormalInTF.y * modNormalInTF.y);
    if (projLength < 1e-3f)
        return;
    float tiltAngle = std::atan(projLength / modNormalInTF.z);
    float qSin, qCos;
    sincos(tiltAngle / 2, &qSin, &qCos);
    float qX = (-modNormalInTF.y / projLength) * qSin;
    float qY = (modNormalInTF.x / projLength) * qSin;
    float qW = qCos;
    Vector3D modTangentInTF(1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
    Vector3D modBitangentInTF(2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

    Matrix3x3 matTFtoW(frameToModify->tangent, frameToModify->bitangent, frameToModify->normal);
    ReferenceFrame bumpShadingFrame(matTFtoW * modTangentInTF,
                                    matTFtoW * modBitangentInTF,
                                    matTFtoW * modNormalInTF);

    *frameToModify = bumpShadingFrame;
}

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromNormalMap)
(CUtexObject texture, shared::TexDimInfo dimInfo, TexCoord2D texCoord) {
    float4 texValue = sample<float4>(texture, dimInfo, texCoord);
    Normal3D modLocalNormal(texValue.x, texValue.y, texValue.z);
    modLocalNormal = 2.0f * modLocalNormal - Normal3D(1.0f);
    if (dimInfo.isLeftHanded)
        modLocalNormal.y *= -1; // DirectX convention
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromNormalMap);

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromNormalMap2ch)
(CUtexObject texture, shared::TexDimInfo dimInfo, TexCoord2D texCoord) {
    float2 texValue = sample<float2>(texture, dimInfo, texCoord);
    float x = 2.0f * texValue.x - 1.0f;
    float y = 2.0f * texValue.y - 1.0f;
    float z = std::sqrt(1.0f - pow2(x) - pow2(y));
    Normal3D modLocalNormal(x, y, z);
    if (dimInfo.isLeftHanded)
        modLocalNormal.y *= -1; // DirectX convention
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromNormalMap2ch);

RT_CALLABLE_PROGRAM Normal3D RT_DC_NAME(readModifiedNormalFromHeightMap)
(CUtexObject texture, shared::TexDimInfo dimInfo, TexCoord2D texCoord) {
    if (dimInfo.isNonPowerOfTwo && dimInfo.isBCTexture) {
        uint32_t bcWidth = (dimInfo.dimX + 3) / 4 * 4;
        uint32_t bcHeight = (dimInfo.dimY + 3) / 4 * 4;
        texCoord.u *= static_cast<float>(dimInfo.dimX) / bcWidth;
        texCoord.v *= static_cast<float>(dimInfo.dimY) / bcHeight;
    }
    float4 heightValues = tex2Dgather<float4>(texture, texCoord.u, texCoord.v, 0);
    constexpr float coeff = (5.0f / 1024);
    uint32_t width = dimInfo.dimX;
    uint32_t height = dimInfo.dimY;
    float dhdu = (coeff * width) * (heightValues.y - heightValues.x);
    float dhdv = (coeff * height) * (heightValues.x - heightValues.w);
    Normal3D modLocalNormal = normalize(Normal3D(-dhdu, dhdv, 1));
    return modLocalNormal;
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(readModifiedNormalFromHeightMap);



template <typename BSDFType>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, uint32_t* bodyData, shared::BSDFBuildFlags flags);



class LambertBRDF {
    RGBSpectrum m_reflectance;

public:
    CUDA_DEVICE_FUNCTION LambertBRDF() {}
    CUDA_DEVICE_FUNCTION LambertBRDF(const RGBSpectrum &reflectance) :
        m_reflectance(reflectance) {}

    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness) const {
        *diffuseReflectance = m_reflectance;
        *specularReflectance = RGBSpectrum::Zero();
        *roughness = 1.0f;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result) const {
        result->dirLocal = cosineSampleHemisphere(sample.uDir[0], sample.uDir[1]);
        result->dirPDensity = result->dirLocal.z / pi_v<float>;
        if (query.dirLocal.z <= 0.0f)
            result->dirLocal.z *= -1;
        return m_reflectance / pi_v<float>;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateF(const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        if (query.dirLocal.z * vSampled.z > 0)
            return m_reflectance / pi_v<float>;
        else
            return RGBSpectrum::Zero();
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        if (query.dirLocal.z * vSampled.z > 0)
            return std::fabs(vSampled.z) / pi_v<float>;
        else
            return 0.0f;
    }

    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateDHReflectanceEstimate(const shared::BSDFQuery &query) const {
        return m_reflectance;
    }
};

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<LambertBRDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, uint32_t* bodyData, shared::BSDFBuildFlags /*flags*/) {
    auto &mat = reinterpret_cast<const shared::LambertianSurfaceMaterial &>(matData.body);
    float4 reflectance = sample<float4>(
        mat.reflectance, mat.reflectanceDimInfo, texCoord);
    auto &bsdfBody = *reinterpret_cast<LambertBRDF*>(bodyData);
    bsdfBody = LambertBRDF(RGBSpectrum(reflectance.x, reflectance.y, reflectance.z));
}



// DichromaticBRDFのDirectional-Hemispherical Reflectanceを事前計算して
// テクスチャー化した結果をフィッティングする。
// Diffuse、Specular成分はそれぞれ
// - baseColor * diffusePreInt(cosV, roughness)
// - specularF0 * specularPreIntA(cosV, roughness) + (1 - specularF0) * specularPreIntB(cosV, roughness)
// で表される。
// https://www.shadertoy.com/view/WtjfRD
CUDA_DEVICE_FUNCTION CUDA_INLINE void calcFittedPreIntegratedTerms(
    float cosV, float roughness,
    float* diffusePreInt, float* specularPreIntA, float* specularPreIntB) {
        {
            float u = cosV;
            float v = roughness;
            float uu = u * u;
            float uv = u * v;
            float vv = v * v;

            *diffusePreInt = min(max(
                -0.417425f * uu +
                -0.958929f * uv +
                -0.096977f * vv +
                1.050356f * u +
                0.534528f * v +
                0.407112f * 1.0f,
                0.0f), 1.0f);
        }
        {
            float u = std::atan2(roughness, cosV);
            float v = std::sqrt(cosV * cosV + roughness * roughness);
            float uu = u * u;
            float uv = u * v;
            float vv = v * v;

            *specularPreIntA = min(max(
                0.133105f * uu +
                -0.278877f * uv +
                -0.417142f * vv +
                -0.192809f * u +
                0.426076f * v +
                0.996565f * 1.0f,
                0.0f), 1.0f);
            *specularPreIntB = min(max(
                0.055070f * uu +
                -0.163511f * uv +
                1.211598f * vv +
                0.089837f * u +
                -1.956888f * v +
                0.741397f * 1.0f,
                0.0f), 1.0f);
        }
}

#define USE_HEIGHT_CORRELATED_SMITH 1
#define USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS 0
#define USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE 0

class DichromaticBRDF {
    struct GGXMicrofacetDistribution {
        float alpha_g;

        CUDA_DEVICE_FUNCTION float evaluate(const Normal3D &m) const {
            if (m.z <= 0.0f)
                return 0.0f;
            float temp = pow2(m.x) + pow2(m.y) + pow2(m.z * alpha_g);
            return pow2(alpha_g) / (pi_v<float> * pow2(temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateSmithG1(const Vector3D &v, const Normal3D &m) const {
            if (dot(v, m) * v.z <= 0)
                return 0.0f;
            float temp = pow2(alpha_g) * (pow2(v.x) + pow2(v.y)) / pow2(v.z);
            return 2 / (1 + std::sqrt(1 + temp));
        }
        CUDA_DEVICE_FUNCTION float evaluateHeightCorrelatedSmithG(
            const Vector3D &v1, const Vector3D &v2, const Normal3D &m) {
            float alpha_g2_tanTheta2_1 = pow2(alpha_g) * (pow2(v1.x) + pow2(v1.y)) / pow2(v1.z);
            float alpha_g2_tanTheta2_2 = pow2(alpha_g) * (pow2(v2.x) + pow2(v2.y)) / pow2(v2.z);
            float Lambda1 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_1)) / 2;
            float Lambda2 = (-1 + std::sqrt(1 + alpha_g2_tanTheta2_2)) / 2;
            float chi1 = (dot(v1, m) / v1.z) > 0 ? 1 : 0;
            float chi2 = (dot(v2, m) / v2.z) > 0 ? 1 : 0;
            return chi1 * chi2 / (1 + Lambda1 + Lambda2);
        }
        CUDA_DEVICE_FUNCTION float sample(
            const Vector3D &v, float u0, float u1, Normal3D* m, float* mPDensity) const {
            // stretch view
            Vector3D sv = normalize(Vector3D(alpha_g * v.x, alpha_g * v.y, v.z));

            // orthonormal basis
            float distIn2D = std::sqrt(sv.x * sv.x + sv.y * sv.y);
            float recDistIn2D = 1.0f / distIn2D;
            Vector3D T1 = (sv.z < 0.9999f) ?
                Vector3D(sv.y * recDistIn2D, -sv.x * recDistIn2D, 0) :
                Vector3D::Ex();
            Vector3D T2(T1.y * sv.z, -T1.x * sv.z, distIn2D);

            // sample point with polar coordinates (r, phi)
            float a = 1.0f / (1.0f + sv.z);
            float r = std::sqrt(u0);
            float phi = pi_v<float> * ((u1 < a) ? u1 / a : 1 + (u1 - a) / (1.0f - a));
            float sinPhi, cosPhi;
            sincos(phi, &sinPhi, &cosPhi);
            float P1 = r * cosPhi;
            float P2 = r * sinPhi * ((u1 < a) ? 1.0f : sv.z);

            // compute normal
            *m = P1 * T1 + P2 * T2 + std::sqrt(1.0f - P1 * P1 - P2 * P2) * sv;

            // unstretch
            *m = normalize(Normal3D(alpha_g * m->x, alpha_g * m->y, m->z));

            float D = evaluate(*m);
            *mPDensity = evaluateSmithG1(v, *m) * absDot(v, *m) * D / std::fabs(v.z);

            return D;
        }
        CUDA_DEVICE_FUNCTION float evaluatePDF(const Vector3D &v, const Normal3D &m) {
            return evaluateSmithG1(v, m) * absDot(v, m) * evaluate(m) / std::fabs(v.z);
        }
    };

protected:
    RGBSpectrum m_diffuseColor;
    RGBSpectrum m_specularF0Color;
    float m_roughness;

public:
    CUDA_DEVICE_FUNCTION DichromaticBRDF() {}
    CUDA_DEVICE_FUNCTION DichromaticBRDF(
        const RGBSpectrum &diffuseColor, const RGBSpectrum &specularF0Color, float smoothness) {
        m_diffuseColor = diffuseColor;
        m_specularF0Color = specularF0Color;
        m_roughness = 1 - smoothness;
    }

    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness) const {
        *diffuseReflectance = m_diffuseColor;
        *specularReflectance = m_specularF0Color;
        *roughness = m_roughness;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &sample,
        shared::BSDFQueryResult* result) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirL;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        float oneMinusDotVN5 = pow5(1 - dirV.z);

        float diffuseWeight;
        float specularWeight;
        if constexpr (USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS) {
            float diffusePreInt;
            float specularPreIntA, specularPreIntB;
            calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

            diffuseWeight = (m_diffuseColor * diffusePreInt).luminance();
            specularWeight =
                (m_specularF0Color * specularPreIntA +
                 (RGBSpectrum::One() - m_specularF0Color) * specularPreIntB).luminance();
        }
        else {
            float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * query.dirLocal.z * query.dirLocal.z;
            float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            float iBaseColor = m_diffuseColor.luminance() * pow2(expectedDiffuseFresnel) *
                lerp(1.0f, 1.0f / 1.51f, m_roughness);

            float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            float iSpecularF0 = m_specularF0Color.luminance();

            diffuseWeight = iBaseColor;
            specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);
        }
        float sumWeights = diffuseWeight + specularWeight;
        if (sumWeights == 0.0f) {
            result->dirPDensity = 0.0f;
            return RGBSpectrum::Zero();
        }

        float uDir1 = sample.uDir[1];
        float uComponent = uDir1;

        float diffuseDirPDF, specularDirPDF;
        Normal3D m;
        float dotLH;
        float D;
        if (sumWeights * uComponent < diffuseWeight) {
            uDir1 = (sumWeights * uComponent - 0) / diffuseWeight;

            // JP: コサイン分布からサンプルする。
            // EN: sample based on cosine distribution.
            dirL = cosineSampleHemisphere(sample.uDir[0], uDir1);
            diffuseDirPDF = dirL.z / pi_v<float>;

            // JP: 同じ方向サンプルをスペキュラー層からサンプルする確率密度を求める。
            // EN: calculate PDF to generate the sampled direction from the specular layer.
            m = halfVector(dirL, dirV);
            dotLH = min(dot(dirL, m), 1.0f);
            float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

            D = ggx.evaluate(m);
        }
        else {
            uDir1 = (sumWeights * uComponent - diffuseWeight) / specularWeight;

            // JP: スペキュラー層のマイクロファセット分布からサンプルする。
            // EN: sample based on the specular microfacet distribution.
            float mPDF;
            D = ggx.sample(dirV, sample.uDir[0], uDir1, &m, &mPDF);
            float dotVH = min(dot(dirV, m), 1.0f);
            dotLH = dotVH;
            dirL = Vector3D(2 * dotVH * m) - dirV;
            if (dirL.z * dirV.z <= 0) {
                result->dirPDensity = 0.0f;
                return RGBSpectrum::Zero();
            }
            float commonPDFTerm = 1.0f / (4 * dotLH);
            specularDirPDF = commonPDFTerm * mPDF;

            // JP: 同じ方向サンプルをコサイン分布からサンプルする確率密度を求める。
            // EN: calculate PDF to generate the sampled direction from the cosine distribution.
            diffuseDirPDF = dirL.z / pi_v<float>;
        }

        float oneMinusDotLH5 = pow5(1 - dotLH);

        float G;
        if constexpr (USE_HEIGHT_CORRELATED_SMITH)
            G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
        else
            G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
        constexpr float F90 = 1.0f;
        RGBSpectrum F = lerp(m_specularF0Color, RGBSpectrum(F90), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        RGBSpectrum specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = RGBSpectrum::Zero();

        float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        float oneMinusDotLN5 = pow5(1 - dirL.z);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);
        RGBSpectrum diffuseValue = m_diffuseColor *
            (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / pi_v<float>);

        RGBSpectrum ret = diffuseValue + specularValue;

        result->dirLocal = entering ? dirL : -dirL;

        // PDF based on one-sample model MIS.
        result->dirPDensity = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        return ret;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateF(const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        if (vSampled.z * query.dirLocal.z <= 0)
            return RGBSpectrum::Zero();

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? vSampled : -vSampled;

        Normal3D m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);

        float oneMinusDotLH5 = pow5(1 - dotLH);

        float D = ggx.evaluate(m);
        float G;
        if constexpr (USE_HEIGHT_CORRELATED_SMITH)
            G = ggx.evaluateHeightCorrelatedSmithG(dirL, dirV, m);
        else
            G = ggx.evaluateSmithG1(dirL, m) * ggx.evaluateSmithG1(dirV, m);
        constexpr float F90 = 1.0f;
        RGBSpectrum F = lerp(m_specularF0Color, RGBSpectrum(F90), oneMinusDotLH5);

        float microfacetDenom = 4 * dirL.z * dirV.z;
        RGBSpectrum specularValue = F * ((D * G) / microfacetDenom);
        if (G == 0)
            specularValue = RGBSpectrum::Zero();

        float F_D90 = 0.5f * m_roughness + 2 * m_roughness * dotLH * dotLH;
        float oneMinusDotVN5 = pow5(1 - dirV.z);
        float oneMinusDotLN5 = pow5(1 - dirL.z);
        float diffuseFresnelOut = lerp(1.0f, F_D90, oneMinusDotVN5);
        float diffuseFresnelIn = lerp(1.0f, F_D90, oneMinusDotLN5);

        RGBSpectrum diffuseValue = m_diffuseColor *
            (diffuseFresnelOut * diffuseFresnelIn * lerp(1.0f, 1.0f / 1.51f, m_roughness) / pi_v<float>);

        RGBSpectrum ret = diffuseValue + specularValue;

        return ret;
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        GGXMicrofacetDistribution ggx;
        ggx.alpha_g = m_roughness * m_roughness;

        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;
        Vector3D dirL = entering ? vSampled : -vSampled;

        Normal3D m = halfVector(dirL, dirV);
        float dotLH = dot(dirL, m);
        float commonPDFTerm = 1.0f / (4 * dotLH);

        float diffuseWeight;
        float specularWeight;
        if constexpr (USE_FITTED_PRE_INTEGRATION_FOR_WEIGHTS) {
            float diffusePreInt;
            float specularPreIntA, specularPreIntB;
            calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

            diffuseWeight = (m_diffuseColor * diffusePreInt).luminance();
            specularWeight =
                (m_specularF0Color * specularPreIntA +
                 (RGBSpectrum::One() - m_specularF0Color) * specularPreIntB).luminance();
        }
        else {
            float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * query.dirLocal.z * query.dirLocal.z;
            float oneMinusDotVN5 = pow5(1 - dirV.z);
            float expectedDiffuseFresnel = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            float iBaseColor = m_diffuseColor.luminance() * pow2(expectedDiffuseFresnel) *
                lerp(1.0f, 1.0f / 1.51f, m_roughness);

            float expectedOneMinusDotVH5 = pow5(1 - dirV.z);
            float iSpecularF0 = m_specularF0Color.luminance();

            diffuseWeight = iBaseColor;
            specularWeight = lerp(iSpecularF0, 1.0f, expectedOneMinusDotVH5);
        }
        float sumWeights = diffuseWeight + specularWeight;
        if (sumWeights == 0.0f)
            return 0.0f;

        float diffuseDirPDF = dirL.z / pi_v<float>;
        float specularDirPDF = commonPDFTerm * ggx.evaluatePDF(dirV, m);

        float ret = (diffuseDirPDF * diffuseWeight + specularDirPDF * specularWeight) / sumWeights;

        return ret;
    }

    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateDHReflectanceEstimate(const shared::BSDFQuery &query) const {
        bool entering = query.dirLocal.z >= 0.0f;
        Vector3D dirV = entering ? query.dirLocal : -query.dirLocal;

        RGBSpectrum diffuseDHR;
        RGBSpectrum specularDHR;
        if constexpr (USE_FITTED_PRE_INTEGRATION_FOR_DH_REFLECTANCE) {
            float diffusePreInt;
            float specularPreIntA, specularPreIntB;
            calcFittedPreIntegratedTerms(dirV.z, m_roughness, &diffusePreInt, &specularPreIntA, &specularPreIntB);

            diffuseDHR = m_diffuseColor * diffusePreInt;
            specularDHR =
                m_specularF0Color * specularPreIntA +
                (RGBSpectrum::One() - m_specularF0Color) * specularPreIntB;
        }
        else {
            float expectedCosTheta_d = dirV.z;
            float expectedF_D90 = 0.5f * m_roughness + 2 * m_roughness * pow2(expectedCosTheta_d);
            float oneMinusDotVN5 = pow5(1 - dirV.z);
            float expectedDiffFGiven = lerp(1.0f, expectedF_D90, oneMinusDotVN5);
            float expectedDiffFSampled = 1.0f; // ad-hoc
            diffuseDHR = m_diffuseColor *
                expectedDiffFGiven * expectedDiffFSampled * lerp(1.0f, 1.0f / 1.51f, m_roughness);

            //float expectedOneMinusDotVH5 = oneMinusDotVN5;
            // (1 - m_roughness) is an ad-hoc adjustment.
            float expectedOneMinusDotVH5 = pow5(1 - dirV.z) * (1 - m_roughness);

            specularDHR = lerp(m_specularF0Color, RGBSpectrum(1.0f), expectedOneMinusDotVH5);
        }

        return min(diffuseDHR + specularDHR, RGBSpectrum::One());
    }
};

class SimplePBR_BRDF : public DichromaticBRDF {
public:
    CUDA_DEVICE_FUNCTION SimplePBR_BRDF() {}
    CUDA_DEVICE_FUNCTION SimplePBR_BRDF(
        const RGBSpectrum &baseColor, float reflectance, float smoothness, float metallic) {
        m_diffuseColor = baseColor * (1 - metallic);
        m_specularF0Color = RGBSpectrum(0.16f * pow2(reflectance) * (1 - metallic)) + baseColor * metallic;
        m_roughness = 1 - smoothness;
    }
};

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<DichromaticBRDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, uint32_t* bodyData, shared::BSDFBuildFlags flags) {
    auto &mat = reinterpret_cast<const shared::DichromaticSurfaceMaterial &>(matData.body);
    float4 diffuseColor = sample<float4>(
        mat.diffuse,
        mat.diffuseDimInfo,
        texCoord);
    float4 specularF0Color = sample<float4>(
        mat.specular,
        mat.specularDimInfo,
        texCoord);
    float smoothness = sample<float>(
        mat.smoothness,
        mat.smoothnessDimInfo,
        texCoord);
    //bool regularize = (flags & BSDFFlags::Regularize) != 0;
    //if (regularize)
    //    smoothness *= 0.5f;
    auto &bsdfBody = *reinterpret_cast<DichromaticBRDF*>(bodyData);
    bsdfBody = DichromaticBRDF(
        RGBSpectrum(diffuseColor.x, diffuseColor.y, diffuseColor.z),
        RGBSpectrum(specularF0Color.x, specularF0Color.y, specularF0Color.z),
        min(smoothness, 0.999f));
}

template<>
CUDA_DEVICE_FUNCTION CUDA_INLINE void setupBSDFBody<SimplePBR_BRDF>(
    const shared::SurfaceMaterial &matData,
    TexCoord2D texCoord, uint32_t* bodyData, shared::BSDFBuildFlags flags) {
    auto &mat = reinterpret_cast<const shared::SimplePBRSurfaceMaterial &>(matData.body);
    float4 baseColor_opacity = sample<float4>(
        mat.baseColor_opacity,
        mat.baseColor_opacity_dimInfo,
        texCoord);
    float4 occlusion_roughness_metallic = sample<float4>(
        mat.occlusion_roughness_metallic,
        mat.occlusion_roughness_metallic_dimInfo,
        texCoord);
    RGBSpectrum baseColor(baseColor_opacity.x, baseColor_opacity.y, baseColor_opacity.z);
    float smoothness = min(1.0f - occlusion_roughness_metallic.y, 0.999f);
    float metallic = occlusion_roughness_metallic.z;
    //bool regularize = (flags & BSDFFlags::Regularize) != 0;
    //if (regularize)
    //    smoothness *= 0.5f;
    auto &bsdfBody = *reinterpret_cast<SimplePBR_BRDF*>(bodyData);
    bsdfBody = SimplePBR_BRDF(baseColor, 0.5f, smoothness, metallic);
}



#define DEFINE_BSDF_CALLABLES(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(BSDFType ## _getSurfaceParameters)(\
        const uint32_t* data,\
        RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _getSurfaceParameters);\
    RT_CALLABLE_PROGRAM RGBSpectrum RT_DC_NAME(BSDFType ## _sampleF)(\
        const uint32_t* data, const shared::BSDFQuery &query, const shared::BSDFSample &sample,\
        shared::BSDFQueryResult* result) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.sampleF(query, sample, result);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _sampleF);\
    RT_CALLABLE_PROGRAM RGBSpectrum RT_DC_NAME(BSDFType ## _evaluateF)(\
        const uint32_t* data, const shared::BSDFQuery &query, const Vector3D &vSampled) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateF(query, vSampled);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluateF);\
    RT_CALLABLE_PROGRAM float RT_DC_NAME(BSDFType ## _evaluatePDF)(\
        const uint32_t* data, const shared::BSDFQuery &query, const Vector3D &vSampled) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluatePDF(query, vSampled);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluatePDF);\
    RT_CALLABLE_PROGRAM RGBSpectrum RT_DC_NAME(BSDFType ## _evaluateDHReflectanceEstimate)(\
        const uint32_t* data, const shared::BSDFQuery &query) {\
        auto &bsdf = *reinterpret_cast<const BSDFType*>(data);\
        return bsdf.evaluateDHReflectanceEstimate(query);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(BSDFType ## _evaluateDHReflectanceEstimate);

DEFINE_BSDF_CALLABLES(LambertBRDF);
DEFINE_BSDF_CALLABLES(DichromaticBRDF);

#undef DEFINE_SETUP_BSDF_CALLABLE

#define DEFINE_SETUP_BSDF_CALLABLE(BSDFType) \
    RT_CALLABLE_PROGRAM void RT_DC_NAME(setup ## BSDFType)(\
        const shared::SurfaceMaterial &matData,\
        TexCoord2D texCoord, uint32_t* bodyData, shared::BSDFBuildFlags flags) {\
        setupBSDFBody<BSDFType>(matData, texCoord, bodyData, flags);\
    }\
    CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(setup ## BSDFType)

DEFINE_SETUP_BSDF_CALLABLE(LambertBRDF);
DEFINE_SETUP_BSDF_CALLABLE(DichromaticBRDF);
DEFINE_SETUP_BSDF_CALLABLE(SimplePBR_BRDF);

#undef DEFINE_SETUP_BSDF_CALLABLE



CUDA_DEVICE_FUNCTION const shared::BSDFProcedureSet &getBSDFProcedureSet(uint32_t slot);

template <bool isGeneric>
class BSDFTemplate;

template <>
class BSDFTemplate<false> {
    HARD_CODED_BSDF m_bsdf;

public:
    CUDA_DEVICE_FUNCTION void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None) {
        setupBSDFBody<HARD_CODED_BSDF>(matData, texCoord, reinterpret_cast<uint32_t*>(&m_bsdf), flags);
    }
    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness) const {
        return m_bsdf.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result) const {
        RGBSpectrum fsValue = m_bsdf.sampleF(query, smp, result);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        RGBSpectrum fsValue = m_bsdf.evaluateF(query, vSampled);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        return m_bsdf.evaluatePDF(query, vSampled);
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateDHReflectanceEstimate(const shared::BSDFQuery &query) const {
        return m_bsdf.evaluateDHReflectanceEstimate(query);
    }
};

template <>
class BSDFTemplate<true> {
    static constexpr uint32_t NumDwords = 16;
    shared::BSDFGetSurfaceParameters m_getSurfaceParameters;
    shared::BSDFSampleF m_sampleF;
    shared::BSDFEvaluateF m_evaluateF;
    shared::BSDFEvaluatePDF m_evaluatePDF;
    shared::BSDFEvaluateDHReflectanceEstimate m_evaluateDHReflectanceEstimate;
    uint32_t m_data[NumDwords];

public:
    CUDA_DEVICE_FUNCTION void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None) {
        matData.setupBSDFBody(matData, texCoord, m_data, flags);
        const shared::BSDFProcedureSet &procSet = getBSDFProcedureSet(matData.bsdfProcSetSlot);
        m_getSurfaceParameters = procSet.getSurfaceParameters;
        m_sampleF = procSet.sampleF;
        m_evaluateF = procSet.evaluateF;
        m_evaluatePDF = procSet.evaluatePDF;
        m_evaluateDHReflectanceEstimate = procSet.evaluateDHReflectanceEstimate;
    }
    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness) const {
        return m_getSurfaceParameters(m_data, diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result) const {
        RGBSpectrum fsValue = m_sampleF(m_data, query, smp, result);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        RGBSpectrum fsValue = m_evaluateF(m_data, query, vSampled);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        return m_evaluatePDF(m_data, query, vSampled);
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateDHReflectanceEstimate(const shared::BSDFQuery &query) const {
        return m_evaluateDHReflectanceEstimate(m_data, query);
    }
};



class BSDF {
    BSDFTemplate<useGenericBSDF> m_body;

public:
    CUDA_DEVICE_FUNCTION void setup(
        const shared::SurfaceMaterial &matData, const TexCoord2D &texCoord,
        shared::BSDFBuildFlags flags = shared::BSDFBuildFlags::None) {
        m_body.setup(matData, texCoord, flags);
    }
    CUDA_DEVICE_FUNCTION void getSurfaceParameters(
        RGBSpectrum* diffuseReflectance, RGBSpectrum* specularReflectance, float* roughness) const {
        return m_body.getSurfaceParameters(diffuseReflectance, specularReflectance, roughness);
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum sampleF(
        const shared::BSDFQuery &query, const shared::BSDFSample &smp,
        shared::BSDFQueryResult* result) const {
        RGBSpectrum fsValue = m_body.sampleF(query, smp, result);

        float snCorrection = std::fabs(result->dirLocal.z / dot(result->dirLocal, query.geometricNormalLocal));
        if (!isfinite(snCorrection))
            return RGBSpectrum::Zero();

        fsValue *= snCorrection;
        Assert(
            (result->dirPDensity > 0 && fsValue.allNonNegativeFinite()) ||
            result->dirPDensity == 0,
            "sampleBSDF: smp: (%g, %g), qDir: (%g, %g, %g), gNormal: (%g, %g, %g)"
            "rDir: (%g, %g, %g), dirPDF: %g, "
            "snCorrection: %g",
            smp.uDir[0], smp.uDir[1],
            print3d(query.dirLocal), print3d(query.geometricNormalLocal),
            print3d(result->dirLocal), result->dirPDensity,
            snCorrection);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        float snCorrection = std::fabs(vSampled.z / dot(vSampled, query.geometricNormalLocal));
        if (!isfinite(snCorrection))
            return RGBSpectrum::Zero();

        RGBSpectrum fsValue = m_body.evaluateF(query, vSampled);

        fsValue *= snCorrection;
        Assert(
            fsValue.allNonNegativeFinite(),
            "evalBSDF: qDir: (%g, %g, %g), gNormal: (%g, %g, %g), "
            "rDir: (%g, %g, %g), "
            "snCorrection: %g",
            print3d(query.dirLocal), print3d(query.geometricNormalLocal),
            print3d(vSampled),
            snCorrection);
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION float evaluatePDF(
        const shared::BSDFQuery &query, const Vector3D &vSampled) const {
        return m_body.evaluatePDF(query, vSampled);
    }
    CUDA_DEVICE_FUNCTION RGBSpectrum evaluateDHReflectanceEstimate(const shared::BSDFQuery &query) const {
        return m_body.evaluateDHReflectanceEstimate(query);
    }
};

} // namespace rtc8::device



namespace rtc8::shared {

CUDA_DEVICE_FUNCTION RGBSpectrum ImageBasedEnvironmentalLight::sample(
    float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) const {
    float u, v;
    float uvPDF;
    importanceMap.sample(uDir0, uDir1, &u, &v, &uvPDF);
    *phi = 2 * pi_v<float> * u;
    *theta = pi_v<float> * v;

    *dirPDensity = uvPDF / (2 * pow2(pi_v<float>) * std::sin(*theta));

    float4 texValue = tex2DLod<float4>(texObj, u, v, 0.0f);
    RGBSpectrum radiance(texValue.x, texValue.y, texValue.z);

    return radiance;
}

RT_CALLABLE_PROGRAM RGBSpectrum RT_DC_NAME(ImageBasedEnvironmentalLight_sample)(
    const uint32_t* data, float uDir0, float uDir1, float* phi, float* theta, float* dirPDensity) {
    auto &envLight = *reinterpret_cast<const ImageBasedEnvironmentalLight*>(data);
    return envLight.sample(uDir0, uDir1, phi, theta, dirPDensity);
}
CUDA_DECLARE_CALLABLE_PROGRAM_POINTER(ImageBasedEnvironmentalLight_sample);

} // namespace rtc8::shared

#endif // #if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
