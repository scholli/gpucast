###############################################################################
# set sources
###############################################################################
include (findGLFW)

IF (UNIX)
  SET(GLFW_LIBRARY_RELEASE ${GLFW_LIBRARY})
  SET(GLFW_LIBRARY_DEBUG ${GLFW_LIBRARY})
ENDIF (UNIX)


FILE(GLOB EXAMPLE_HELLO_WORLD_GLFW_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
PROJECT(${_EXE_NAME})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
                     ${GLFW_INCLUDE_DIR}
                     ${GLEW_INCLUDE_DIR}
)

SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${BOOST_LIBRARY_DIRS} 
                 ${QHULL_LIBRARY_DIRS} 
                 ${FREEIMAGE_LIBRARY_DIRS}
                 ${GLFW_LIBRARY_DIRS}
)

LINK_DIRECTORIES (${LIBRARY_DIRS})

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_HELLO_WORLD_GLFW_SRC}
)

SET_TARGET_PROPERTIES( ${_EXE_NAME} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

TARGET_LINK_LIBRARIES(${_EXE_NAME} 
                      gpucast_gl 
                      gpucast_core
		                  general ${GLUT_glut_LIBRARY}
                      general ${GLEW_LIBRARY}     
                      optimized ${GLFW_LIBRARY_RELEASE} debug ${GLFW_LIBRARY_DEBUG}     
		                  ${FREEIMAGE_LIBRARY}
		                  ${FREEIMAGEPLUS_LIBRARY}
)


###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
copy_external_runtime_libraries(glfw glew freeimage)
apply_debug_working_directory()