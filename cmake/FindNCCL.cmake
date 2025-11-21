# cmake/FindNCCL.cmake
# Custom NCCL finder module for CMake

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS /usr/include /usr/local/include /opt/nccl/include
    NO_DEFAULT_PATH
)

find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS /usr/lib /usr/local/lib /opt/nccl/lib /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
    add_library(NCCL::NCCL INTERFACE IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${NCCL_LIBRARY}"
    )
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
