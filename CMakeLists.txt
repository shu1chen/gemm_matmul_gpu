cmake_minimum_required(VERSION 3.5)
project(gemm_matmul_gpu)

option(WITH_MKL "Include Intel MKL" ON)

execute_process(COMMAND git submodule update --init --recursive --no-fetch
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(DNNL_BUILD_TESTS OFF CACHE BOOL "Disable dnnl tests")
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Disable dnnl examples")
set(DNNL_ENABLE_JIT_PROFILING OFF CACHE INTERNAL "" FORCE)
set(DNNL_CPU_RUNTIME SEQ CACHE INTERNAL "" FORCE) # Disable OPENMP
set(DNNL_BLAS_VENDOR MKL CACHE INTERNAL "" FORCE) # Use MKL cblas

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wno-deprecated-anon-enum-enum-conversion")

add_subdirectory(libs/oneDNN)

include_directories(libs/oneDNN/include)

configure_file(${CMAKE_SOURCE_DIR}/gemm_matmul_gpu.sh ${CMAKE_BINARY_DIR}/gemm_matmul_gpu COPYONLY)

set(LIBRARIES
  dnnl
  xetla
  )

if(WITH_MKL)
  find_path(MKL_ROOT include/mkl.h
    PATHS $ENV{MKLROOT} /opt/intel/oneapi/mkl/latest
    DOC "Path to MKL root directory")

  # Find MKL includes.
  find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include/)
  if(MKL_INCLUDE_DIR)
    message(STATUS "Found MKL include directory: ${MKL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "MKL include directory not found")
  endif()

  # Find MKL libraries.
  find_library(MKL_CORE_LIBRARY NAMES mkl_core PATHS ${MKL_ROOT}/lib/intel64)
  if(MKL_CORE_LIBRARY)
    get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIBRARY} DIRECTORY)
    message(STATUS "Found MKL library directory: ${MKL_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "MKL library directory not found")
  endif()

  include_directories(${MKL_INCLUDE_DIR})
  add_definitions(-DWITH_MKL -DMKL_ILP64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  set(MKL_LIBRARIES
    ${MKL_LIBRARY_DIR}/libmkl_core.a
    ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a
    ${MKL_LIBRARY_DIR}/libmkl_intel_ilp64.a
    )
  list(APPEND LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)
  list(APPEND LIBRARIES gomp)
endif()

add_executable(bench bench.cpp)
target_link_libraries(bench ${LIBRARIES})
