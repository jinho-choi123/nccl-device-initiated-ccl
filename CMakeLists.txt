# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(device-initiated-nccl)

# Directory for Module Finders
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#### Setup compiler ####

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#### Setup CUDA Compiler(NVCC) ####
# Find NVCC to set CMAKE_CUDA_COMPILER
find_package(NVCC REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_COMPUTE_CAPABILITY 89)

# Enable CUDA
enable_language(CUDA)

#### Setup required dependencies ####

# Find NCCL
find_package(NCCL REQUIRED)

if(NCCL_FOUND)
    message(STATUS "NCCL include directory: ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL library: ${NCCL_LIBRARY}")
else()
    message(FATAL_ERROR "NCCL not found!")
endif()

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

# Add common/ subdirectory
add_subdirectory(common/)

# Add lib/ subdirectory
add_subdirectory(lib/)

