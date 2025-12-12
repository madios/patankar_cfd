# FVM/tests/CMakeLists.txt

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(tests_run
        FVM_modelEquations_test.cpp
        KERNEL_linearEquationSolver_test.cpp
        KERNEL_performance_test.cpp
)
include_directories(/usr/local/include)

# Link with GoogleTest and our static library
target_link_libraries(tests_run
        PRIVATE
        gtest_main
        numerics_kernel_lib
        mesh_lib
)

# Include GoogleTest
include(GoogleTest)
gtest_discover_tests(tests_run)
