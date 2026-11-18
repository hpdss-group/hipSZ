# Test and measurement programs. Each upstream-style test covers one
# (type, dim) pair across all three encoding modes on 512M elements;
# edge is the boundary regression suite; stress hammers the decoupled
# lookback; compat checks the pre-merge 1D public API; bench is the
# dual-metric performance harness (installed, not a ctest entry).
# Plain C++ host code; hip/hip_runtime.h is host-compatible and hip::host
# propagates the include paths and runtime.

set(install_dir ${PROJECT_BINARY_DIR}/tests/bin)
set(hipSZ_test_f32_1D "hipSZ_test_f32_1D")
set(hipSZ_test_f64_1D "hipSZ_test_f64_1D")
set(hipSZ_test_f32_2D "hipSZ_test_f32_2D")
set(hipSZ_test_f64_2D "hipSZ_test_f64_2D")
set(hipSZ_test_f32_3D "hipSZ_test_f32_3D")
set(hipSZ_test_f64_3D "hipSZ_test_f64_3D")
set(hipSZ_test_edge "hipSZ_test_edge")
set(hipSZ_test_stress "hipSZ_test_stress")
set(hipSZ_compat_test "hipSZ_compat_test")
set(hipSZ_bench "hipSZ_bench")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(${hipSZ_test_f32_1D} hipSZ_test_f32_1D.cpp)
add_executable(${hipSZ_test_f64_1D} hipSZ_test_f64_1D.cpp)
add_executable(${hipSZ_test_f32_2D} hipSZ_test_f32_2D.cpp)
add_executable(${hipSZ_test_f64_2D} hipSZ_test_f64_2D.cpp)
add_executable(${hipSZ_test_f32_3D} hipSZ_test_f32_3D.cpp)
add_executable(${hipSZ_test_f64_3D} hipSZ_test_f64_3D.cpp)
add_executable(${hipSZ_test_edge} hipSZ_test_edge.cpp)
add_executable(${hipSZ_test_stress} hipSZ_test_stress.cpp)
add_executable(${hipSZ_compat_test} hipSZ_compat_test.cpp)
add_executable(${hipSZ_bench} hipSZ_bench.cpp)

set(hipSZ_TESTS
    ${hipSZ_test_f32_1D} ${hipSZ_test_f64_1D}
    ${hipSZ_test_f32_2D} ${hipSZ_test_f64_2D}
    ${hipSZ_test_f32_3D} ${hipSZ_test_f64_3D}
    ${hipSZ_test_edge} ${hipSZ_test_stress}
    ${hipSZ_compat_test}
)

foreach(tgt ${hipSZ_TESTS} ${hipSZ_bench})
    target_link_libraries(${tgt} PRIVATE hipSZ_static hip::host)
    set_target_properties(${tgt} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
    install(TARGETS ${tgt} DESTINATION bin)
endforeach()

foreach(tgt ${hipSZ_TESTS})
    add_test(NAME ${tgt} COMMAND $<TARGET_FILE:${tgt}>)
endforeach()

# The 512M-element suites and the stress loop can run for a while on a busy
# device; a timeout turns a lookback hang into a CI-visible failure.
set_tests_properties(
    ${hipSZ_test_f32_1D} ${hipSZ_test_f64_1D}
    ${hipSZ_test_f32_2D} ${hipSZ_test_f64_2D}
    ${hipSZ_test_f32_3D} ${hipSZ_test_f64_3D}
    PROPERTIES TIMEOUT 1800
)
set_tests_properties(${hipSZ_test_stress} PROPERTIES TIMEOUT 900)
set_tests_properties(${hipSZ_test_edge} ${hipSZ_compat_test} PROPERTIES TIMEOUT 600)
