cmake_minimum_required(VERSION 3.15)

project(cheby)

set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory("externals/pybind11")
pybind11_add_module(cheby "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp")
target_compile_definitions(cheby PUBLIC -DPYBIND11)
