cmake_minimum_required(VERSION 3.0)

project(cheby)

set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cheby.cpp")

if (MSVC)
    list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()

if (CHEBY_STATIC_LIBRARY)
    add_library(cheby STATIC ${SOURCES})
    target_include_directories(cheby PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
else()
    if (NOT CHEBY_NO_PYBIND11)
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11")
        pybind11_add_module(cheby "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp" ${SOURCES})
        target_compile_definitions(cheby PUBLIC -DPYBIND11)
    endif()
endif()
