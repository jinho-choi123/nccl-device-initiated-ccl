
#include "kernels.cuh"
#include "utils.cuh"
#include "utils.h"
#include <nccl.h>
#include <stdio.h>

void *allReduce(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  // Standard NCCL communicator initialization(same as Host API)
  if (my_rank == 0) {
    printf("Starting Device API LSA-All-Reduce Initialization\n");
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID in case of MPI
  util_broadcast(0, my_rank, &nccl_unique_id);

  // Set device context for this rank
  CUDACHECK(cudaSetDevice(local_device));

  printf("Rank %d using GPU device %d\n", my_rank, local_device);

  // Initialize NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
  printf("Rank %d initialized NCCL communicator for %d total ranks\n", my_rank, total_ranks);

  // Allocate memory for AllReduce operation
  size_t count = 1024 * 1024; // 1M elements
  size_t size_bytes = count * sizeof(float);

  float *host_data = (float *)malloc(size_bytes);
  void *d_sendbuff;
  void *d_recvbuff;

  // Initialize host data with rank-specific values for verification
  for (size_t i = 0; i < count; i++) {
    host_data[i] = (float)my_rank;
  }

  CUDACHECK(cudaMemcpy(d_sendbuff, host_data, size_bytes, cudaMemcpyHostToDevice));
  printf("Rank %d initialized data with value %d\n", my_rank, my_rank);

  // Create stream for kernel execution
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  if (my_rank == 0) {
    printf("Starting AllReduce with %zu elements (%zu MB) using Device API\n", count, size_bytes / (1024 * 1024));
    printf("Expected result: sum of ranks 0 to %d = %d per element\n", total_ranks - 1,
           (total_ranks * (total_ranks - 1)) / 2);
  }

  // Call All-reduce CCL
  NCCLCHECK(ncclAllReduce(d_sendbuff, d_recvbuff, count, ncclFloat, ncclSum, comm, stream));

  // Wait for completion - kernel performs AllReduce.
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("  Rank %d completed AllReduce kernel execution\n", my_rank);

  // Verify results by copying back and checking
  CUDACHECK(cudaMemcpy(host_data, d_recvbuff, size_bytes, cudaMemcpyDeviceToHost));
  float expected = (float)((total_ranks * (total_ranks - 1)) / 2);
  bool success = true;
  for (int i = 0; i < count; i++) {
    if (host_data[i] != expected) {
      success = false;
      break;
    }
  }

  if (my_rank == 0) {
    printf("AllReduce completed. Result verification: %s\n", success ? "PASSED" : "FAILED");
    if (success) {
      printf("All elements correctly sum to %.0f (ranks 0-%d)\n", expected, total_ranks - 1);
    }
  }

  // Cleanup
  free(host_data);

  // Device API specific cleanup
  NCCLCHECK(ncclMemFree(d_sendbuff));
  NCCLCHECK(ncclMemFree(d_recvbuff));

  // Standard NCCL cleanup
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  return NULL;
}

int main(int argc, char **argv) { run_example(argc, argv, allReduce); }
