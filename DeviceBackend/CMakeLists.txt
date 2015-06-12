cmake_minimum_required (VERSION 3.0)

########################################
####        Generic Part            ####
########################################

project( DeviceBackend )
set( OUTPUT_NAME DeviceBackend )

########################################
####	        Src files           ####
########################################

set( sources
	DeviceBackend.cu
)

set( headers
	DeviceBackend.cu.h
)

#runtime libs for cuda
set( cuda_deps cudadevrt )
if (WIN32)
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
else ()
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

########################################
# Miscellaneous parallel computing lib #
########################################
#Change device execution for fun !
#set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)
#set(THRUST_DEVICE_SYSTEM "THRUST_DEVICE_SYSTEM_OMP -Xcompiler -fopenmp" )
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB)

list( APPEND CUDA_NVCC_FLAGS -DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM} )

########################################
####           TBB Part             ####
########################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "FindTBB.cmake")
# Execute the FindTBB macro
find_package(TBB)

if(NOT TBB_FOUND)
    MESSAGE (STATUS "TBB not found. TBB implementation will not be supported.")
else(NOT TBB_FOUND)
	set( tbb_deps ${_tbb_deps} )
endif(NOT TBB_FOUND)

########################################
####         	OMP Part	        ####
########################################
set(omp_deps gomp)

########################################
####	       Executable           ####
########################################

cuda_add_executable( ${OUTPUT_NAME} ${sources} )

target_link_libraries( ${OUTPUT_NAME}
${cuda_deps}
${omp_deps}
${tbb_deps}
)

set_property( TARGET ${OUTPUT_NAME} PROPERTY OUTPUT_NAME ${output_name} )

install( TARGETS ${OUTPUT_NAME} RUNTIME DESTINATION ${RUNTIME_DEST} )