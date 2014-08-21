###############################################################################
# set sources
###############################################################################
include (find_glfw)

FILE(GLOB EXAMPLE_HELLO_WORLD_GLFW_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${GLFW_INCLUDE_DIRS}
                     ${GLEW_INCLUDE_DIRS}
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
                     ${GLFW_INCLUDE_DIR}
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

IF (MSVC)
TARGET_LINK_LIBRARIES(${_EXE_NAME} gpucast_gl gpucast_core
		     debug ${FREEGLUT_LIBRARY_DEBUG} optimized ${FREEGLUT_LIBRARY_RELEASE} )
ELSE (MSVC)
TARGET_LINK_LIBRARIES(${_EXE_NAME} 
		      gpucast_gl 
	              gpucast_core
		      ${FREEGLUT_LIBRARY}
		      ${GLFW_LIBRARY}
		      ${FREEIMAGE_LIBRARY}
		      ${FREEIMAGEPLUS_LIBRARY}
		      boost_filesystem
		      ${GLEW_LIBRARY}
		      X11 Xxf86vm Xrandr pthread m Xi GL GLU
                     )
ENDIF (MSVC)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
IF (MSVC)
    copy_runtime_dependencies ( ${_EXE_NAME} \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    FOREACH (_LIB_DIR ${LIBRARY_DIRS})
        copy_runtime_dependencies ( ${_EXE_NAME} \"${_LIB_DIR}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    ENDFOREACH (_LIB_DIR ${LIB_PATHS})
ENDIF (MSVC)
