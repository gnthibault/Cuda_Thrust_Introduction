# Locate Intel Threading Building Blocks include paths and libraries
# FindTBB.cmake can be found at https://code.google.com/p/findtbb/
# Written by Hannes Hofmann <hannes.hofmann _at_ informatik.uni-erlangen.de>
# Improvements by Gino van den Bergen <gino _at_ dtecta.com>,
#   Florian Uhlig <F.Uhlig _at_ gsi.de>,
#   Jiri Marsik <jiri.marsik89 _at_ gmail.com>
# The MIT License
# Copyright (c) 2011 Hannes Hofmann

# This module defines
# TBB_FOUND
#TBB_INSTALL_DIR
#TBB_INCLUDE_DIR
#TBB_LIBRARY_DIR
#_tbb_deps 
#_tbb_deps_debug

# OS architecture detection
if ( WIN32 ) # Windows
	if ( CMAKE_CL_64 )
	    message ( "-- TBB: 64 bits architecture detected for the target" )
	    set ( TBB_ARCHITECTURE "intel64" )
	else ()
	    message( "-- TBB: 32 bits architecture detected for the target" )
	    set ( TBB_ARCHITECTURE "ia32" )
	endif ()
else () # Linux ( How to manage host and target with different architectures ??? )
	if ( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	    message ( "-- TBB: 64 bits architecture detected for the target" )
	    set ( TBB_ARCHITECTURE "intel64" )
	else ()
	    message ( "-- TBB: 32 bits architecture detected for the target" )
	    set ( TBB_ARCHITECTURE "ia32" )
	endif ()	
endif ()
set(_TBB_ARCHITECTURE ${TBB_ARCHITECTURE})

set(_TBB_LIB_NAME "tbb")
set(_TBB_LIB_MALLOC_NAME "${_TBB_LIB_NAME}malloc")
set(_TBB_LIB_DEBUG_NAME "${_TBB_LIB_NAME}_debug")
set(_TBB_LIB_MALLOC_DEBUG_NAME "${_TBB_LIB_MALLOC_NAME}_debug")

if (WIN32)
    set(_TBB_DEFAULT_INSTALL_DIR "C:/Program Files/Intel/TBB" "C:/Program Files (x86)/Intel/TBB")

    if (MSVC71)
        set (_TBB_COMPILER "vc7.1")
    endif(MSVC71)
    if (MSVC80)
        set(_TBB_COMPILER "vc8")
    endif(MSVC80)
    if (MSVC90)
        set(_TBB_COMPILER "vc9")
    endif(MSVC90)
    if(MSVC10)
        set(_TBB_COMPILER "vc10")
    endif(MSVC10)
    if(MSVC11)
        set(_TBB_COMPILER "vc11")
    endif(MSVC11)
    if(MSVC12)
        set(_TBB_COMPILER "vc12")
    endif(MSVC12)
endif (WIN32)

if (UNIX)
	# LINUX
	set(_TBB_DEFAULT_INSTALL_DIR "/opt/intel/tbb" "/usr/local/tbb" $ENV{TBB_ROOT})
	set( TBB_COMPILER "gcc4.4" )
	set(_TBB_COMPILER ${TBB_COMPILER})
endif (UNIX)

#-- Clear the public variables
set (TBB_FOUND "NO")
#Try to find path automatically
find_path(_TBB_INSTALL_DIR
	name "include/tbb/tbb.h"
	PATHS ${_TBB_DEFAULT_INSTALL_DIR} $ENV{TBB_INSTALL_DIR} ENV CPATH NO_DEFAULT_PATH )

# sanity check
if (NOT _TBB_INSTALL_DIR)

	message ("ERROR: Unable to find Intel TBB install directory. ${_TBB_INSTALL_DIR}")
	
	if (TBB_FIND_REQUIRED)
		message(FATAL_ERROR "Could NOT find TBB library.")
	endif (TBB_FIND_REQUIRED)

else (NOT _TBB_INSTALL_DIR)

	set (TBB_FOUND "YES")
	message(STATUS "--Found Intel TBB")

	#-- Look for include directory and set ${TBB_INCLUDE_DIR}
	set( TBB_INCLUDE_DIR "${_TBB_INSTALL_DIR}/include" )

	#-- Look for libraries
	set ( TBB_LIBRARY_DIR "${_TBB_INSTALL_DIR}/lib/${_TBB_ARCHITECTURE}/${_TBB_COMPILER}")

	include_directories ( ${TBB_INCLUDE_DIR} )
	link_directories ( ${TBB_LIBRARY_DIR} )
      
	#release
	set ( _tbb_deps 
		${_TBB_LIB_NAME}
		${_TBB_LIB_MALLOC_NAME} )
	#debug
	set ( _tbb_deps_debug
		${_TBB_LIB_DEBUG_NAME}
		${_TBB_LIB_MALLOC_DEBUG_NAME} )

 	#-------------------------------------------------------------------
        # TBB REDIST
        #-------------------------------------------------------------------
       
	if(WIN32) # Windows
	install (DIRECTORY ${TBB_LIBRARY_DIR}/
	    DESTINATION ${RUNTIME_DEST}
	    COMPONENT core
	    FILES_MATCHING
	    PATTERN "*${_TBB_LIB_NAME}.lib"
	    PATTERN "*${_TBB_LIB_MALLOC_NAME}.lib"
	    PATTERN "*${_TBB_LIB_DEBUG_NAME}.lib"
	    PATTERN "*${_TBB_LIB_MALLOC_DEBUG_NAME}.lib" 
	    )
	else()	# Linux
	install (DIRECTORY ${TBB_LIBRARY_DIR}/
	    DESTINATION ${RUNTIME_DEST}
	    COMPONENT core
	    FILES_MATCHING 
	    PATTERN "*${_TBB_LIB_NAME}*.so*" 
	    PATTERN "*${_TBB_LIB_MALLOC_NAME}*.so*" 
	    PATTERN "*${_TBB_LIB_DEBUG_NAME}*.so*" 
	    PATTERN "*${_TBB_LIB_MALLOC_DEBUG_NAME}*.so*" 
	    )
	endif()

        #-------------------------------------------------------------------
        # END TBB REDIST
        #-------------------------------------------------------------------

endif (NOT _TBB_INSTALL_DIR)
