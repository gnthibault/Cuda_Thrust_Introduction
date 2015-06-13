cmake_minimum_required(VERSION 3.0)

project (Cuda_Thrust_Introduction)
set( RUNTIME_DEST ${CMAKE_CURRENT_BINARY_DIR}/bin )

########################################
####  Generic Compilation options   ####
########################################

#Compiler must support c++11
ADD_DEFINITIONS( ${CMAKE_CXX_FLAGS} "-std=c++11" )

if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
	if(UNIX)
		ADD_DEFINITIONS(${CMAKE_CXX_FLAGS} "-Wall -O0 -ggdb")
	endif()
else()
	if(UNIX)
		ADD_DEFINITIONS(${CMAKE_CXX_FLAGS} "-O3 -mtune=native -march=native")
	endif()
endif()

########################################
####    Cuda Compilation options    ####
########################################

find_package(CUDA 7.0 REQUIRED)

#generic compilation flag, defines target architecture, type of build, ...
list( APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -std=c++11)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	list( APPEND CUDA_NVCC_FLAGS --device-debug --generate-line-info -g -G -DTHRUST_DEBUG )
else ()
	list( APPEND CUDA_NVCC_FLAGS -use_fast_math -O3 ) #--default-stream per-thread )
endif()

########################################
#  Adding various sample directories   #
########################################

add_subdirectory( HostDeviceVector )
add_subdirectory( DeviceBackend )
add_subdirectory( AsynchronousLaunch )
add_subdirectory( MultiGpuThrust )
#add_subdirectory( ThrustVectorWrappingCublas )

