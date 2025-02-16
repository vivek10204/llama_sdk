cmake_minimum_required(VERSION 3.22)

# Define the project
project(MyProjectName)

# Set CMake flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_COMMIT=unknown -DBUILD_COMPILER=unknown -DBUILD_TARGET=Android")

# Add subdirectory with explicit source and binary directories
add_subdirectory(${CMAKE_SOURCE_DIR}/src/llama_cpp ${CMAKE_BINARY_DIR}/llama_cpp_build)
