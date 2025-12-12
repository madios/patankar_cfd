# projectroot/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(Patankar_CFD VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

add_subdirectory(modules/FVM)
add_subdirectory(modules/Mesh)
add_subdirectory(modules/NumericsKernel)
add_subdirectory(test_modules)
