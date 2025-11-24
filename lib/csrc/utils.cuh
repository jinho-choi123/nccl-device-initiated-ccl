#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
// Error checking
#define NCCLCHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    ncclResult_t res = cmd;                                                                                            \
    if (res != ncclSuccess) {                                                                                          \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res));                 \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                                                            \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CUDACHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));                 \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                                                            \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth, props.maxThreadsPerBlock,
         props.maxThreadsPerMultiProcessor, props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024, props.sharedMemPerMultiprocessor / 1024,
         props.totalConstMem / 1024, props.multiProcessorCount, props.warpSize);
};
