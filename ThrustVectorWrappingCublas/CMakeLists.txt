cmake_minimum_required (VERSION 3.0)

########################################
####        Generic Part            ####
########################################

project( ThrustVectorWrappingCublas )
set( OUTPUT_NAME ThrustVectorWrappingCublas )

########################################
####	        Src files           ####
########################################

set( sources
	main.cu
)

set( headers
	CublasWrapper.cu.h
	Optimisation.cu.h
	ThrustWrapper.cu.h
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
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)
#set(THRUST_DEVICE_SYSTEM "THRUST_DEVICE_SYSTEM_OMP -Xcompiler -fopenmp")
#set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB)

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
####	         GSL Part           ####
########################################

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "FindGSL.cmake") 
FIND_PACKAGE(GSL)

if( GSL_FOUND )
	add_definitions( -DUSE_GSL )
	if( GSL_CBLAS_FOUND )
		add_definitions( -DUSE_GSL_CBLAS )
		set( blas_deps gslcblas )
	endif()
	INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIRS})
	LINK_DIRECTORIES(${GSL_LIBRARY_DIRS})
	set( gsl_deps gsl )
endif()

########################################
####       	    MKL Part            ####
########################################

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "FindMKL.cmake") 
set( MKL_MULTI_THREADED 1 )
#set( INTEL_ROOT "/softs/intel" )
#set( INTEL_ROOT "/opt/intel/composer_xe_2015.0.090" )
FIND_PACKAGE(MKL)
if( MKL_FOUND )
	set( blas_deps ${MKL_LIBRARIES} )
	LINK_DIRECTORIES(${MKL_LIBRARY_PATH})
	unset(omp_deps)
endif()

########################################
####	       Executable           ####
########################################

cuda_add_executable( ${OUTPUT_NAME} ${sources} )
CUDA_ADD_CUBLAS_TO_TARGET( ${OUTPUT_NAME} )

target_link_libraries( ${OUTPUT_NAME}
${cuda_deps}
${omp_deps}
${tbb_deps}
${gsl_deps}
${blas_deps} )

set_property( TARGET ${OUTPUT_NAME} PROPERTY OUTPUT_NAME ${output_name} )

install( TARGETS ${OUTPUT_NAME} RUNTIME DESTINATION ${RUNTIME_DEST} )