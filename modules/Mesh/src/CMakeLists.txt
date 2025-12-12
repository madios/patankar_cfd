cmake_minimum_required(VERSION 3.14)
project(MeshLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mesh_lib STATIC
        structured2d.cpp
)

target_include_directories(mesh_lib
        PUBLIC
        ../include   # hvis du har en include-mappe
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/modules/Common
        ${CMAKE_SOURCE_DIR}/modules/NumericsKernel/include
)

# If you use compile-time scalar selection, propagate it publicly:
# Example: choose float or double at configure time
# target_compile_definitions(numerics_kernel PUBLIC KERNEL_SCALAR_T=float)

