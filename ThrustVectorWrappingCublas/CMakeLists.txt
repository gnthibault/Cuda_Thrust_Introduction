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
	ThrustVector.cu
)

set( headers
	ThrustVector.cu.h
)

#runtime libs for cuda
set( cuda_deps cudadevrt )
if (WIN32)
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
else ()
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

########################################
####	       Executable           ####
########################################

cuda_add_executable( ${OUTPUT_NAME} ${sources} )
CUDA_ADD_CUBLAS_TO_TARGET( ${OUTPUT_NAME} )

target_link_libraries( ${OUTPUT_NAME} ${cuda_deps} )

set_property( TARGET ${OUTPUT_NAME} PROPERTY OUTPUT_NAME ${output_name} )

install( TARGETS ${OUTPUT_NAME} RUNTIME DESTINATION ${RUNTIME_DEST} )