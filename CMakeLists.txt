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

################################################################
# Configure and find libraries
################################################################
SET(GLOBAL_EXT_DIR ${CMAKE_SOURCE_DIR}/externals)

if (UNIX)
	pkg_check_modules(GL REQUIRED gl)
endif (UNIX)

include(macros)
include(find_compiler)
include(find_boost)
include(find_glew)
include(find_freeglut)
include(find_freeimage)
include(find_qhull)

IF (MSVC)
	set(ENV{CUDA_PATH} ${GLOBAL_EXT_DIR}/CUDA6.5)
	set(ENV{CUDA_BIN_PATH} ${GLOBAL_EXT_DIR}/CUDA6.5/bin)
	include(findCUDA)
ELSEIF (UNIX)
	set(ENV{CUDA_PATH} /opt/cuda/latest/cuda/)
	set(ENV{CUDA_BIN_PATH} /opt/cuda/latest/cuda/bin)
	include(FindCUDA)
ENDIF (MSVC)

################################################################
# Write Configuration
################################################################
SET(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
  ${CMAKE_CURRENT_SOURCE_DIR}/gpucast_core/include/gpucast/core/config.hpp
)

set(LIB_PATHS
    ${BOOST_LIBRARY_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

set(INCLUDE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BOOST_INCLUDE_DIRS}
)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

################################################################
# preprocessor configuration
################################################################
IF (UNIX)
	set(CMAKE_CXX_FLAGS_RELEASE "-s -O4 -fPIC --std=c++0x")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -fPIC --std=c++0x")
ELSEIF(MSVC)
	set(CMAKE_CXX_FLAGS_RELEASE "-D NDEBUG -D _SECURE_SCL=0 -D _SCL_SECURE_NO_WARNINGS -D _CRT_SECURE_NO_DEPRECATE/MD /MP")
	set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG /MDd /Zi /MP")
  add_definitions(-D _USE_MATH_DEFINES -D NOMINMAX)
ENDIF(UNIX)

################################################################
# gpucast Configuration
################################################################

################################################################
# Create build flags
################################################################
SET (BUILD_FLAGS " ")

################################################################
# Create libraries
################################################################
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

ADD_SUBDIRECTORY(gpucast_math)
ADD_SUBDIRECTORY(gpucast_gl)
ADD_SUBDIRECTORY(gpucast_glut)
ADD_SUBDIRECTORY(gpucast_core)
ADD_SUBDIRECTORY(gpucast_volume)
ADD_SUBDIRECTORY(examples)

################################################################
# Summary
################################################################

message(STATUS "" )
message(STATUS "Summary:" )
message(STATUS " build type: ${CMAKE_BUILD_TYPE}" )
message(STATUS "" )

print_dependency(BOOST)
print_dependency(GLEW)
print_dependency(QHULL)
print_dependency(FREEIMAGE)
print_dependency(FREEGLUT)
