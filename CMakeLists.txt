################################################################
# GPUcast
################################################################
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
PROJECT(GPUCAST CXX)

# version number
set(GPUCAST_MAJOR 1)
set(GPUCAST_MINOR 0)
set(GPUCAST_DESCRIPTION "GPUCast - library for rendering higher order primitives")

# cmake minimum version 2.8.0
cmake_minimum_required(VERSION 2.8.0)

if (UNIX)
	find_package(PkgConfig)
endif(UNIX)

# Location where cmake first looks for modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Default install
                 prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

################################################################
# gpucast Configuration
################################################################
SET(ENABLE_GPUCAST_VOLUME_MODULE false CACHE BOOL "Build module gpucast volume")

################################################################
# Configure and find dependencies
################################################################
SET(GLOBAL_EXT_DIR ${CMAKE_SOURCE_DIR}/externals)

if (UNIX)
	pkg_check_modules(GL REQUIRED gl) 
endif (UNIX)

include(macros)

##boost#########################################################
set(BOOST_INCLUDEDIR ${GLOBAL_EXT_DIR}/boost/include)
set(BOOST_LIBRARYDIR ${GLOBAL_EXT_DIR}/boost/lib)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS ON)
find_package( Boost 1.54.0 REQUIRED thread regex system program_options timer chrono filesystem date_time log)
ADD_DEFINITIONS("-DBOOST_ALL_NO_LIB")

##glut##########################################################
set(GLUT_ROOT_PATH ${GLOBAL_EXT_DIR}/freeglut)
find_package(GLUT REQUIRED)

##glew##########################################################
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${GLOBAL_EXT_DIR}/glew/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${GLOBAL_EXT_DIR}/glew/lib)
include(findGLEW)

##freeimage#####################################################
include(findFreeImage)

##qhull#########################################################
include(findQHull)

##cuda##########################################################
IF (${ENABLE_GPUCAST_VOLUME_MODULE})
  IF (MSVC)
	  set(ENV{CUDA_PATH} ${GLOBAL_EXT_DIR}/CUDA)
	  set(ENV{CUDA_BIN_PATH} ${GLOBAL_EXT_DIR}/CUDA/bin)
	  include(findCUDA)
  ELSE (MSVC)
	  set(ENV{CUDA_PATH} /opt/cuda/latest/cuda/)
	  set(ENV{CUDA_BIN_PATH} /opt/cuda/latest/cuda/bin)
	  include(FindCUDA)
  ENDIF (MSVC)
ENDIF (${ENABLE_GPUCAST_VOLUME_MODULE})

################################################################
# Write Configuration
################################################################
#FILE(GLOB_RECURSE _RUNTIME_LIBRARIES ${GLOBAL_EXT_DIR} ${GLOBAL_EXT_DIR}/*.dll)
#SET(GPUCAST_RUNTIME_LIBRARIES ${_RUNTIME_LIBRARIES} CACHE INTERNAL "Runtime libraries.")

################################################################
# Write Configuration
################################################################
IF (MSVC)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
ELSEIF (UNIX)
  SET(CMAKE_INSTALL_PREFIX /opt/local)
ENDIF (MSVC)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
  ${CMAKE_CURRENT_SOURCE_DIR}/gpucast_core/include/gpucast/core/config.hpp
)

set(LIB_PATHS
    ${BOOST_LIBRARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

set(INCLUDE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

################################################################
# preprocessor configuration
################################################################
IF (UNIX)
	set(CMAKE_CXX_FLAGS_RELEASE "-s -O4 -fPIC -std=c++1y")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -fPIC -std=c++1y")
ELSEIF(MSVC)
	set(CMAKE_CXX_FLAGS_RELEASE "-D NDEBUG -D _SECURE_SCL=0 -D _SCL_SECURE_NO_WARNINGS -D _CRT_SECURE_NO_DEPRECATE /MD /MP")
	set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG /MDd /Zi /MP")
  add_definitions(-D _USE_MATH_DEFINES -D NOMINMAX)
ENDIF(UNIX)

################################################################
# Create build flags
################################################################
SET (BUILD_FLAGS " ")

################################################################
# Create libraries
################################################################
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

ADD_SUBDIRECTORY(gpucast_math)
ADD_SUBDIRECTORY(gpucast_gl)
ADD_SUBDIRECTORY(gpucast_glut)
ADD_SUBDIRECTORY(gpucast_core)

IF (${ENABLE_GPUCAST_VOLUME_MODULE})
ADD_SUBDIRECTORY(gpucast_volume)
ENDIF (${ENABLE_GPUCAST_VOLUME_MODULE})

ADD_SUBDIRECTORY(examples)

################################################################
# Summary
################################################################

message(STATUS "" )
message(STATUS "Summary:" )
message(STATUS " build type: ${CMAKE_BUILD_TYPE}" )
message(STATUS "" )
