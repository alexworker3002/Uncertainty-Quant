cmake_minimum_required(VERSION 3.18)
project(dmt_implicit_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(dmt_implicit_ext src/bindings.cpp src/dmt_implicit.cpp)
target_include_directories(dmt_implicit_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
