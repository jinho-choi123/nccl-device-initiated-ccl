set(CSRC_DIR csrc)
set(INCLUDE_DIR include)

file(GLOB_RECURSE CSRC_FILES ${CSRC_DIR}/*.cpp)

add_executable(main ${CSRC_FILES})
target_include_directories(main PRIVATE ${INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(main PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES})