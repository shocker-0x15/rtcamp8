set(TARGET_NAME "denoiser")

file(
    GLOB_RECURSE SELECTIVE_COMMON_SOURCES
    "../common/*"
    "../ext/fpng/src/fpng.h"
    "../ext/fpng/src/fpng.cpp"
    "../ext/tinyexr/*"
)
list(
    FILTER SELECTIVE_COMMON_SOURCES INCLUDE REGEX
    ".*\.(h|hpp|cuh|c|cc|cpp|cu|vert|frag)$")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../common/shaders/.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../common/utils/optix_util.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../common/utils/optix.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../common/utils/gl_util.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../common/common_renderer.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/deps/miniz/examples/.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/deps/ZFP/.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/(examples|experimental|test)/.*")
list(
    FILTER SELECTIVE_COMMON_SOURCES EXCLUDE REGEX
    "../ext/tinyexr/test_tinyexr.cc$")
file(GLOB STB_SOURCES "../ext/stb_*")
set(
    SELECTIVE_COMMON_SOURCES
    ${SELECTIVE_COMMON_SOURCES}
    ${STB_SOURCES}
)

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_SOURCES
    "network_interface.cu"
)

set(
    GPU_KERNEL_DEPENDENCIES
    "../common/utils/cuda_util.h"
    "../common/utils/optix_util.h"
    "../common/utils/optixu_on_cudau.h"
    "../common/common_shared.h"
    "../common/basic_types.h"
    "${TARGET_NAME}_shared.h"
)

# non-essentials
source_group(
    "non-essentials/utils" REGULAR_EXPRESSION
    "../common/utils/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials" REGULAR_EXPRESSION
    "../common/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/tinyexr" REGULAR_EXPRESSION
    "../ext/tinyexr/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/fpng" REGULAR_EXPRESSION
    "../ext/fpng/src/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext/tinyexr/miniz" REGULAR_EXPRESSION
    "../ext/tinyexr/deps/miniz/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "non-essentials/ext" REGULAR_EXPRESSION
    "../ext/[^/]*\.(h|hpp|c|cc|cpp)$")
source_group(
    "non-essentials/GPU kernels" REGULAR_EXPRESSION
    "../common/gpu_kernels/.*")
source_group(
    "non-essentials/shaders" REGULAR_EXPRESSION
    "../common/shaders/.*")

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "essentials" FILES
    ${CUDA_SOURCES})
source_group(
    "essentials/GPU kernels" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernels/[^/]*\.(h|cu)$")

list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)
list(APPEND DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})

list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "$<$<CONFIG:Debug>:-G>")

add_executable(
    "${TARGET_NAME}"
    ${SELECTIVE_COMMON_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${CUDA_SOURCES}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF CUDA_EXTENSIOINS OFF)
target_compile_definitions("${TARGET_NAME}" PRIVATE ${DEFINITIONS})
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
    "CUDA_UTIL_DONT_USE_GL_INTEROP"
)
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"

    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/tinyexr"
    "../ext/tinyexr/deps/miniz"
    "../ext/tiny-cuda-nn/dependencies"
    "../ext/tiny-cuda-nn/include"
)
target_link_libraries(
    "${TARGET_NAME}"
    CUDA::cuda_driver
    tiny-cuda-nn
)
    