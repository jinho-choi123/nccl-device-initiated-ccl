# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(device-initiated-nccl)

#### Setup compiler ####

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#### Setup required dependencies ####
# Directory for Module Finders
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Find NCCL
find_package(NCCL REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find MPI
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI C++ compiler: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI C++ include path: ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI C++ libraries: ${MPI_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "MPI not found!")
endif()

# Add lib/ subdirectory
add_subdirectory(lib/)
