
#include "kernels.cuh"
#include "utils.cuh"
#include "utils.h"
#include <nccl.h>
#include <stdio.h>

#define NCCL_DEVICE_CTA_COUNT 16
#define NCCL_DEVICE_THREADS_PER_CTA 512

void *allReduce(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Device API LSA-All-Reduce Initialization\n");
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  util_broadcast(0, my_rank, &nccl_unique_id);
}

int main(int argc, char **argv) {
  CudaDeviceInfo();

  int *N = (int *)malloc(sizeof(int));
  *N = 1;

  int *d_N = nullptr;
  CUDACHECK(cudaMalloc(&d_N, sizeof(int)));
  CUDACHECK(cudaMemcpy(d_N, N, sizeof(int), cudaMemcpyHostToDevice));

  // lsa_all_reduce<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA>>>(d_N);

  CUDACHECK(cudaMemcpy(N, d_N, sizeof(int), cudaMemcpyDeviceToHost));

  printf("N: %d\n", *N);

  CUDACHECK(cudaFree(d_N));
  free(N);
}
