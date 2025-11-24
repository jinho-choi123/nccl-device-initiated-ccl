find_program(CMAKE_CUDA_COMPILER
  NAMES nvcc
  PATHS
    /usr/local/cuda/bin
    /usr/local/cuda-12.8/bin
    /usr/local/cuda-12.6/bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVCC DEFAULT_MSG CMAKE_CUDA_COMPILER)
