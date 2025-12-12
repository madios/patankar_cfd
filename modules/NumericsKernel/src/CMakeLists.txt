#  NumericsKernel/src/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(NumericsKernel_Lib VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()



# Create static library for the linear solver
add_library(numerics_kernel_lib STATIC
        KERNEL.cpp
        LinEqsSolvers.cpp
)


# Set include directories for the library
target_include_directories(numerics_kernel_lib
        PUBLIC
        ../include
        ${CMAKE_SOURCE_DIR}/modules/Common
        PRIVATE
        .
)



# If you use compile-time scalar selection, propagate it publicly:
# Example: choose float or double at configure time
# target_compile_definitions(numerics_kernel PUBLIC KERNEL_SCALAR_T=float)


# ==========================
# BLAS/LAPACK via Accelerate
# --------------------------
# We enable LAPACK in Blaze and link Apple Accelerate which provides
# ==========================
# NOTE: We need to link against BLAS/LAPACK (e.g. Accelerate on macOS) because blaze::gesv calls dgesv_
include_directories(/usr/local/include)

# Find the Accelerate framework (macOS system BLAS/LAPACK)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Tell Blaze that LAPACK is available and use the CBLAS header from Accelerate
# (ACCELERATE_NEW_LAPACK and BLAZE_BLAS_INCLUDE_FILE are typical for macOS)
target_compile_definitions(numerics_kernel_lib PRIVATE
    BLAZE_USE_LAPACK=1
    ACCELERATE_NEW_LAPACK=1
    BLAZE_BLAS_INCLUDE_FILE="cblas.h"
)

# Link the library with Accelerate so symbols like dgesv_ resolve
target_link_libraries(numerics_kernel_lib PUBLIC ${ACCELERATE_FRAMEWORK})
