include (find_qt4)

###############################################################################
# set sources
###############################################################################
file(GLOB EXAMPLE_TRIMVIEW_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp include/*.hpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${QT_INCLUDE_DIRS}
                     ${GLEW_INCLUDE_DIRS}
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
)

SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${BOOST_LIBRARY_DIRS} 
                 ${QHULL_LIBRARY_DIRS} 
                 ${FREEIMAGE_LIBRARY_DIRS}
		             ${QT_LIBRARY_DIRS}
)

LINK_DIRECTORIES (${LIBRARY_DIRS})

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_TRIMVIEW_SRC}
)

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

IF (MSVC)
    TARGET_LINK_LIBRARIES(${_EXE_NAME} ${LIBRARIES} gpucast_gl gpucast_core 
                          optimized QtGui4.lib
                          optimized QtOpenGL4.lib
                          optimized QtCore4.lib
                          optimized qtmain.lib
                          debug QtGuid4.lib
                          debug QtOpenGLd4.lib
                          debug QtCored4.lib
                          debug qtmaind.lib
    )
ELSEIF (UNIX)
    MESSAGE("linking libraries for example_igs_trimcurve_viewer.out not implemented")
ENDIF(MSVC)


###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
IF (MSVC)
    copy_runtime_dependencies ( ${_EXE_NAME} \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    FOREACH (_LIB_DIR ${LIB_PATHS})
        copy_runtime_dependencies ( ${_EXE_NAME} \"${_LIB_DIR}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    ENDFOREACH (_LIB_DIR ${LIB_PATHS})
ENDIF (MSVC)