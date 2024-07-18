cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(SobelWithCudaConvolution LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
# find_package(cuda-api-wrappers 0.5.1 REQUIRED)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

enable_language(CUDA)
# Add source files
set(SOURCES
    src/main.cpp
    src/conv2d.cu
)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

add_executable(SobelWithCudaConvolution ${SOURCES})


target_compile_features(SobelWithCudaConvolution PUBLIC cxx_std_14)


# Include directories
target_include_directories(SobelWithCudaConvolution PRIVATE include)

# Link CUDA
target_link_libraries(SobelWithCudaConvolution ${CUDA_LIBRARIES}  ${OpenCV_LIBS})

# Additional compile options for verbose output
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --verbose -G -g")


# #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_35")
# # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --verbose -G -g")

# add_executable(SobelWithCudaConvolution ${SOURCES})

# target_compile_features(SobelWithCudaConvolution  PUBLIC cxx_std_14)

# # set_target_properties( SobelWithCudaConvolution 
#                     #    PROPERTIES  CUDA_ARCHITECTURES "35") #CUDA_SEPARABLE_COMPILATION ON

# target_link_libraries(SobelWithCudaConvolution  ${OpenCV_LIBS} ${CUDAToolkit_LIBRARIES})
# # set_property(TARGET SobelWithCudaConvolution PROPERTY CUDA_VERBOSE_BUILD ON)