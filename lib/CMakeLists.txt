
set(CSRC_DIR csrc)
set(INCLUDE_DIR include)

file(GLOB_RECURSE CSRC_FILES ${CSRC_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_FILES ${CSRC_DIR}/*.cu)

# Build run_add_one
set(RUN_ADD_ONE_SOURCE ${CSRC_DIR}/run_add_one.cu)
add_executable(run_add_one ${RUN_ADD_ONE_SOURCE})
target_include_directories(run_add_one PRIVATE ${INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH} ${NCCL_INCLUDE_DIR})
target_link_libraries(run_add_one PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES} common)

# Build run_all_reduce
set(RUN_ALL_REDUCE_SOURCE ${CSRC_DIR}/run_all_reduce.cu)
add_executable(run_all_reduce ${RUN_ALL_REDUCE_SOURCE})
target_include_directories(run_all_reduce PRIVATE ${INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH} ${NCCL_INCLUDE_DIR})
target_link_libraries(run_all_reduce PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES} common)

# Build run_lsa_all_reduce
set(RUN_LSA_ALL_REDUCE_SOURCE ${CSRC_DIR}/run_lsa_all_reduce.cu)
add_executable(run_lsa_all_reduce ${RUN_LSA_ALL_REDUCE_SOURCE})
target_include_directories(run_lsa_all_reduce PRIVATE ${INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH} ${NCCL_INCLUDE_DIR})
target_link_libraries(run_lsa_all_reduce PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES} common)

# Build main
set(MAIN_SOURCE ${CSRC_DIR}/main.cu)
add_executable(main ${MAIN_SOURCE})
target_include_directories(main PRIVATE ${INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH} ${NCCL_INCLUDE_DIR})
target_link_libraries(main PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES} common)
