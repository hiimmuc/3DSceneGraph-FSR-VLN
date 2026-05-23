cmake_minimum_required(VERSION 3.10)
project(fast_gicp)

cmake_policy(SET CMP0074 NEW)
option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    # 32-bit ARM optimizations with NEON support
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops") #-flto
  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops") 
  add_definitions(-DX86_ARCH)
endif()
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# -------------------------
# Main library
# -------------------------
add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
if(OPENMP_FOUND AND TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
endif()

# -------------------------
# CUDA VGICP
# -------------------------
if(BUILD_VGICP_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
  add_definitions(-DUSE_VGICP_CUDA)
  set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
  # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -std=c++17 -Xcompiler=-fPIC -Xcompiler=-march=armv8-a+fp16 -Xcompiler=-mcpu=native")

  cuda_add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
  )
  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
    thirdparty/nvbio
  )
  cuda_add_cublas_to_target(fast_vgicp_cuda)

  # 添加 CUDA 相关源到主库
  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
    src/fast_gicp/ndt/ndt_cuda.cpp
  )
  target_link_libraries(fast_gicp fast_vgicp_cuda)
  add_dependencies(fast_gicp fast_vgicp_cuda)
endif()


# -------------------------
# Export include & library for parent project
# -------------------------
# 上层 CMake 可以使用 target_link_libraries(parent_target fast_gicp)
target_include_directories(fast_gicp PUBLIC include)


