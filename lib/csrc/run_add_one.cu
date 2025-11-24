
#include "kernels.cuh"
#include "utils.cuh"
#include <stdio.h>

int main(int argc, char **argv) {
  int *N = (int *)malloc(sizeof(int));
  *N = 1;

  printf("Initial N: %d\n", *N);

  int *d_N = nullptr;
  CUDACHECK(cudaMalloc(&d_N, sizeof(int)));
  CUDACHECK(cudaMemcpy(d_N, N, sizeof(int), cudaMemcpyHostToDevice));

  add_one<<<1, 1>>>(d_N);

  CUDACHECK(cudaMemcpy(N, d_N, sizeof(int), cudaMemcpyDeviceToHost));

  printf("Final N: %d\n", *N);

  CUDACHECK(cudaFree(d_N));
  free(N);
}
