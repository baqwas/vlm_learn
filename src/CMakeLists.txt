cmake_minimum_required(VERSION 3.12)
project(vlm_engine_project)

find_package(OpenMP REQUIRED)

execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")

find_package(pybind11 REQUIRED)

pybind11_add_module(vlm_engine engine.cpp)
target_link_libraries(vlm_engine PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(vlm_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)
