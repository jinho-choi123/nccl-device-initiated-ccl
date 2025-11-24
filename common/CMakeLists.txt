
set(CSRC_DIR csrc)
set(INCLUDE_DIR include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPI_SUPPORT")

file(GLOB_RECURSE CSRC_FILES ${CSRC_DIR}/*.cpp)

add_library(common SHARED ${CSRC_FILES})
target_include_directories(common PUBLIC ${INCLUDE_DIR} ${NCCL_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(common PUBLIC CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES})
