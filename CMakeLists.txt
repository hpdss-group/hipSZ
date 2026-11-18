# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(hipSZ VERSION 1.0.0 DESCRIPTION "Error-bounded GPU lossy compression library (HIP port)")

# Enable languages
enable_language(CXX HIP)

# Find required HIP toolkit
find_package(hip REQUIRED)

# Set C++ and HIP standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Architecture precedence. An explicitly passed -DCMAKE_HIP_ARCHITECTURES
# wins (command-line -D entries are stored with cache type UNINITIALIZED).
# CMake's own detection stores a STRING list of host architectures; the
# project default is gfx926, the Hygon K100 target documented by the main
# hipSZ README. HIPSZ_HIP_ARCHITECTURES can also be set directly. The
# imported engine was validated on gfx936; pass
# -DCMAKE_HIP_ARCHITECTURES=gfx936 when configuring for that device.
get_property(_hipsz_arch_type CACHE CMAKE_HIP_ARCHITECTURES PROPERTY TYPE)
if(_hipsz_arch_type STREQUAL "UNINITIALIZED")
    # User-provided architecture: keep it.
elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(HIPSZ_HIP_ARCHITECTURES "gfx926" CACHE STRING "HIP architecture targeted by hipSZ")
    set(CMAKE_HIP_ARCHITECTURES ${HIPSZ_HIP_ARCHITECTURES})
else()
    # CMake auto-detected host architectures; use the documented default.
    set(HIPSZ_HIP_ARCHITECTURES "gfx926" CACHE STRING "HIP architecture targeted by hipSZ")
    set(CMAKE_HIP_ARCHITECTURES ${HIPSZ_HIP_ARCHITECTURES} CACHE STRING "HIP architectures to build for" FORCE)
endif()
unset(_hipsz_arch_type)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Static archives must be position-independent: downstream executables link
# as PIE by default on this toolchain.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# HIP_ENABLE_WARP_SYNC_BUILTINS enables the __shfl_*_sync family on DTK HIP;
# the kernels pass an explicit 64-bit mask plus width=32 (logical warp).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-unsequenced -D__HIP_PLATFORM_AMD__")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wno-unused-result -Wno-unsequenced -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS")

# The quantization helper relies on unmulated multiply/add steps; contraction
# would shift the quantization boundary and break the error-bound contract.
# Keep it off unless explicitly re-enabled for experiments.
option(HIPSZ_FP_CONTRACT "Allow FMA contraction in device code (changes quantization boundaries)" OFF)
if(NOT HIPSZ_FP_CONTRACT)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffp-contract=off")
endif()

# List of source files
set(hipSZ_SOURCES
    src/hipSZ_kernels_1D_f32.hip
    src/hipSZ_kernels_1D_f64.hip
    src/hipSZ_kernels_2D_f32.hip
    src/hipSZ_kernels_2D_f64.hip
    src/hipSZ_kernels_3D_f32.hip
    src/hipSZ_kernels_3D_f64.hip
    src/hipSZ_utility.hip
    src/hipSZ_timer.hip
    src/hipSZ_entry_1D_f32.hip
    src/hipSZ_entry_1D_f64.hip
    src/hipSZ_entry_2D_f32.hip
    src/hipSZ_entry_2D_f64.hip
    src/hipSZ_entry_3D_f32.hip
    src/hipSZ_entry_3D_f64.hip
    src/hipSZ.hip
    src/hipSZ_compat.hip
)

# Create static and shared libraries
add_library(hipSZ_static STATIC ${hipSZ_SOURCES})
add_library(hipSZ_shared SHARED ${hipSZ_SOURCES})

# Include directories for both libraries
target_include_directories(hipSZ_static
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/hipSZ>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(hipSZ_shared
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/hipSZ>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link HIP runtime. The public headers expose HIP types, so consumers must
# compile with __HIP_PLATFORM_AMD__ as well; propagate it through the targets.
target_link_libraries(hipSZ_static PRIVATE hip::host)
target_link_libraries(hipSZ_shared PRIVATE hip::host)
target_compile_definitions(hipSZ_static PUBLIC __HIP_PLATFORM_AMD__)
target_compile_definitions(hipSZ_shared PUBLIC __HIP_PLATFORM_AMD__)

# Set output name for libraries
set_target_properties(hipSZ_static PROPERTIES OUTPUT_NAME hipSZ)
set_target_properties(hipSZ_shared PROPERTIES OUTPUT_NAME hipSZ)

# Headers for installation
set(public_headers
    include/hipSZ.h
)

# Enable CTest before adding the example/test subdirectories (add_test() in a
# subdirectory requires enable_testing() in its parent scope). Tests run the
# build-tree executables directly, so no prior installation is required for
# `ctest` to be meaningful.
enable_testing()

# Optionally build examples and tests
option(hipSZ_BUILD_EXAMPLES "Option to enable building the example programs" ON)
if (hipSZ_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

option(hipSZ_BUILD_TESTS "Option to enable building the test programs" ON)
if (hipSZ_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

# Include the Installing.cmake script
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Installing)
