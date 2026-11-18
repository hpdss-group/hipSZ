# Standalone consumer project: verifies that an installed hipSZ package can
# be found and linked from a fresh CMake project, for both library flavors.
#
#   cmake -S consumer -B consumer/build \
#     -DCMAKE_HIP_COMPILER=/opt/dtk-26.04/llvm/bin/clang++ \
#     -DCMAKE_PREFIX_PATH=<hipSZ install dir>
cmake_minimum_required(VERSION 3.21)
project(hipSZ_consumer LANGUAGES CXX)

find_package(hipSZ CONFIG REQUIRED)

add_executable(consumer_static consumer_main.cpp)
target_link_libraries(consumer_static PRIVATE hipSZ::hipSZ_static hip::host)

add_executable(consumer_shared consumer_main.cpp)
target_link_libraries(consumer_shared PRIVATE hipSZ::hipSZ_shared hip::host)

enable_testing()
add_test(NAME consumer_static COMMAND consumer_static)
add_test(NAME consumer_shared COMMAND consumer_shared)
