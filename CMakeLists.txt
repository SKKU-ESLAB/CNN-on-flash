cmake_minimum_required(VERSION 3.0.0)
project(acl-blas-on-flash)

set(N_IO_THR 4 CACHE STRING "")
set(N_COMPUTE_THR 4 CACHE STRING "")
set(PROGRAM_BUDGET 3145728 CACHE STRING "") # Bytes
set(MAX_SIMUL_REQS 4096 CACHE STRING "")
set(MAX_STRIDES 4096 CACHE STRING "")
set(MAX_EVENTS 4096 CACHE STRING "")
set(SECTOR_LEN 512 CACHE STRING "")
set(IS_ALIGNED IS_512_ALIGNED CACHE STRING "")
set(GEMM_BLK_SIZE 256 CACHE STRING "") # the number of row and column of submatrices
set(OMP_CHUNK_SIZE 32768 CACHE STRING "")
set(MAX_NNZS 10000000 CACHE STRING "")
set(MAP_BLK_SIZE 1048576 CACHE STRING "")
set(OVERLAP_CHECK TRUE CACHE STRING "")

add_definitions(-DN_IO_THR=${N_IO_THR}
                -DN_COMPUTE_THR=${N_COMPUTE_THR}
                -DPROGRAM_BUDGET=${PROGRAM_BUDGET}
                -DMAX_SIMUL_REQS=${MAX_SIMUL_REQS}
                -DMAX_STRIDES=${MAX_STRIDES}
                -DMAX_EVENTS=${MAX_EVENTS}
                -DSECTOR_LEN=${SECTOR_LEN}
                -DIS_ALIGNED=${IS_ALIGNED}
                -DGEMM_BLK_SIZE=${GEMM_BLK_SIZE}
                -DOMP_CHUNK_SIZE=${OMP_CHUNK_SIZE}
                -DMAX_NNZS=${MAX_NNZS}
                -DMAP_BLK_SIZE=${MAP_BLK_SIZE}
                -DOVERLAP_CHECK=${OVERLAP_CHECK}
								-D_32BIT_MACHINE=1
                )

# Compiler options
set(CXX g++)

# CXX Config
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
add_compile_options(-std=c++14 -Wall ${CONFIG_FLAGS} -fopenmp -Wfatal-errors)

#ACL Config
set(ACL_ROOT [arm_compute_library_path])
include_directories(include ${ACL_ROOT} ${ACL_ROOT}/include)
link_libraries(arm_compute arm_compute_core pthread OpenCL)
link_directories(${ACL_ROOT}/build)

# Other config
add_compile_options(${OPTIM_CXXFLAGS})
link_libraries(aio streamline_annotate)

# Generate library
file(GLOB FHANDLES "src/file_handles/*.cpp")
file(GLOB BLAS "src/blas/gemm.cpp")
file(GLOB SCHED "src/scheduler/*.cpp")
add_library(fblas STATIC ${FHANDLES} ${BLAS} ${SCHED} src/lib_funcs.cpp src/utils.cpp)
link_libraries(fblas gomp)

# Generate drivers
add_executable(in_mem_gemm_driver drivers/in_mem_gemm.cpp)
add_executable(gemm_driver drivers/gemm.cpp /usr/lib/arm-linux-gnueabihf/pthread_atfork.o)
add_executable(print print.cpp)
# Generate tests/misc related stuff
add_executable(dense_create misc/dense_create.cpp misc/gen_common.h)

add_custom_command(TARGET fblas PRE_BUILD COMMAND clang-format-4.0 -i ../include/*.h ../include/*/*.h ../src/*.cpp ../src/*/*.cpp ../misc/*.cpp ../drivers/*.cpp)
