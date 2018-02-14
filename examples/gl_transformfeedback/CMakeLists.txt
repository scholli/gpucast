###############################################################################
# set sources
###############################################################################
FILE(GLOB EXAMPLE_HELLO_WORLD_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
PROJECT(${_EXE_NAME})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_glut/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
                     ${GLUT_INCLUDE_DIR}
                     ${GLEW_INCLUDE_DIR}
)

SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${QHULL_LIBRARY_DIRS} 
                 ${FREEIMAGE_LIBRARY_DIRS}
                 ${FREEGLUT_LIBRARY_DIRS}
)

LINK_DIRECTORIES (${LIBRARY_DIRS})

add_definitions(-D FREEGLUT_LIB_PRAGMAS=0)

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_HELLO_WORLD_SRC}
)

SET_TARGET_PROPERTIES( ${_EXE_NAME} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_glut gpucast_core)

TARGET_LINK_LIBRARIES(${_EXE_NAME} 
                      gpucast_gl 
                      gpucast_glut
                      gpucast_core
		                  general ${GLUT_glut_LIBRARY} 
                      debug ${FREEIMAGE_LIBRARY_DEBUG} optimized ${FREEIMAGE_LIBRARY}
                      debug ${FREEIMAGE_PLUS_LIBRARY_DEBUG} optimized ${FREEIMAGE_PLUS_LIBRARY}
)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
copy_external_runtime_libraries(freeglut glew freeimage)
apply_debug_working_directory()