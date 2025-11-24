#pragma once
#include <cuda_runtime.h>

__global__ void add_one(int *N) { *N += 1; };
