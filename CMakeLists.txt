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

find_package(pybind11 CONFIG QUIET)

if (NOT pybind11_FOUND)
    if (NOT DEFINED PYTHON_EXECUTABLE)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
    endif()

    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if (pybind11_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
        find_package(pybind11 CONFIG QUIET)
    endif()
endif()

if (NOT pybind11_FOUND)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11/CMakeLists.txt")
        add_subdirectory("externals/pybind11")
    else()
        message(FATAL_ERROR "pybind11 was not found via CMake package discovery and vendored fallback is unavailable. Install pybind11 (pip/conda/system) or initialize submodules.")
    endif()
endif()

pybind11_add_module(cheby "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp")
target_compile_definitions(cheby PUBLIC PYBIND11)
target_include_directories(cheby PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (CHEBY_USE_VENDORED_EIGEN)
    target_include_directories(cheby PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
else()
    target_link_libraries(cheby PRIVATE Eigen3::Eigen)
endif()

cmake_print_variables(CMAKE_CXX_FLAGS)
