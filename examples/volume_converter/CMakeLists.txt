IF (${ENABLE_GPUCAST_VOLUME_MODULE})

###############################################################################
# set sources
###############################################################################
file(GLOB EXAMPLE_VOLUME_CONVERTER_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.hpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_glut/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
                     ${CMAKE_SOURCE_DIR}/gpucast_volume/include
                     ${FREEGLUT_INCLUDE_DIR}
                     ${GLEW_INCLUDE_DIR}
)

###############################################################################
# dependencies
###############################################################################
SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${BOOST_LIBRARY_DIRS} 
                 ${QHULL_LIBRARY_DIRS} 
                 ${FREEIMAGE_LIBRARY_DIRS}
                 ${FREEGLUT_LIBRARY_DIRS}
)

LINK_DIRECTORIES (${LIBRARY_DIRS})

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_VOLUME_CONVERTER_SRC}
)

ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

IF (MSVC)
  TARGET_LINK_LIBRARIES(${_EXE_NAME} 
                        gpucast_gl 
                        gpucast_glut 
                        gpucast_core                           
                        gpucast_volume   
                        )
ELSEIF (UNIX)
  TARGET_LINK_LIBRARIES(${_EXE_NAME} 
                        gpucast_gl 
                        gpucast_glut 
                        gpucast_core                           
                        gpucast_volume   
                        ${FREEGLUT_LIBRARIES}                        
                        ${FREEIMAGE_LIBRARIES}
                        )
ENDIF (MSVC)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
IF (MSVC)
    #copy_runtime_dependencies ( ${_EXE_NAME} \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    copy_runtime_dependencies ( ${_EXE_NAME} \"${GLOBAL_EXT_DIR}/Qt5/bin\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    FOREACH (_LIB_DIR ${LIB_PATHS})
        copy_runtime_dependencies ( ${_EXE_NAME} \"${_LIB_DIR}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    ENDFOREACH (_LIB_DIR ${LIB_PATHS})
ENDIF (MSVC)

ENDIF (${ENABLE_GPUCAST_VOLUME_MODULE})