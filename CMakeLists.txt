cmake_minimum_required(VERSION 3.15)

project(cheby)

set(CMAKE_CXX_STANDARD 17)

include(CMakePrintHelpers)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
    cmake_print_variables(CMAKE_CXX_FLAGS)
endif()

find_package(Eigen3 CONFIG QUIET)

if (NOT Eigen3_FOUND)
    find_package(Eigen3 QUIET)
endif()

set(CHEBY_USE_VENDORED_EIGEN OFF)
if (NOT Eigen3_FOUND)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/Eigen/Core")
        set(CHEBY_USE_VENDORED_EIGEN ON)
    else()
        message(FATAL_ERROR "Eigen3 was not found via CMake package discovery and vendored fallback is unavailable. Install Eigen3 (system/conda/vcpkg) or initialize submodules.")
    endif()
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if (NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if (nanobind_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
endif()

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(cheby "${CMAKE_CURRENT_SOURCE_DIR}/src/nanobind_wrapper.cpp")
target_include_directories(cheby PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (CHEBY_USE_VENDORED_EIGEN)
    target_include_directories(cheby PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
else()
    target_link_libraries(cheby PRIVATE Eigen3::Eigen)
endif()

cmake_print_variables(CMAKE_CXX_FLAGS)
