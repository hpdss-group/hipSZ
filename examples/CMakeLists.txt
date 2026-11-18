# Example programs: the hipSZ command-line compressor/decompressor and the
# two pre-merge 1D round-trip examples linked from the README.
#
# The CLI is plain C++ host code; hip/hip_runtime.h is host-compatible and
# hip::host propagates the include paths and runtime.
find_package(hip REQUIRED)

set(install_dir ${PROJECT_BINARY_DIR}/examples/bin)
set(hipSZ "hipSZ")
set(hipSZ_test_f32 "hipSZ_test_f32")
set(hipSZ_test_f64 "hipSZ_test_f64")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(${hipSZ} hipSZ.cpp)
add_executable(${hipSZ_test_f32} hipSZ_test_f32.cpp)
add_executable(${hipSZ_test_f64} hipSZ_test_f64.cpp)

target_link_libraries(${hipSZ} PRIVATE hipSZ_static hip::host)
target_link_libraries(${hipSZ_test_f32} PRIVATE hipSZ_static hip::host)
target_link_libraries(${hipSZ_test_f64} PRIVATE hipSZ_static hip::host)

set_target_properties(${hipSZ} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${hipSZ_test_f32} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${hipSZ_test_f64} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

install(TARGETS ${hipSZ} DESTINATION bin)
install(TARGETS ${hipSZ_test_f32} DESTINATION bin)
install(TARGETS ${hipSZ_test_f64} DESTINATION bin)

# Keep the README's `ctest` entry points: the legacy 1D examples still run
# here and exercise the compatibility layer end to end.
add_test(NAME hipSZTest_f32 COMMAND $<TARGET_FILE:${hipSZ_test_f32}>)
add_test(NAME hipSZTest_f64 COMMAND $<TARGET_FILE:${hipSZ_test_f64}>)
set_tests_properties(hipSZTest_f32 hipSZTest_f64 PROPERTIES TIMEOUT 1800)
