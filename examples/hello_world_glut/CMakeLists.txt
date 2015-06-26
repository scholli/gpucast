###############################################################################
# set sources
###############################################################################
FILE(GLOB EXAMPLE_HELLO_WORLD_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_glut/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
                     ${FREEGLUT_INCLUDE_DIR}
                     ${GLEW_INCLUDE_DIR}
)

SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${BOOST_LIBRARY_DIRS} 
                 ${QHULL_LIBRARY_DIRS} 
                 ${FREEIMAGE_LIBRARY_DIRS}
                 ${FREEGLUT_LIBRARY_DIRS}
)

LINK_DIRECTORIES (${LIBRARY_DIRS})

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_HELLO_WORLD_SRC}
)

SET_TARGET_PROPERTIES( ${_EXE_NAME} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_glut gpucast_core)

IF (MSVC)
TARGET_LINK_LIBRARIES(${_EXE_NAME} 
                      gpucast_gl 
                      gpucast_glut
                      gpucast_core
		                  debug ${FREEGLUT_LIBRARY_DEBUG} optimized ${FREEGLUT_LIBRARY_RELEASE} 
                      debug ${FREEIMAGE_LIBRARY_DEBUG} optimized ${FREEIMAGE_LIBRARY_RELEASE}
                      debug ${FREEIMAGEPLUS_LIBRARY_DEBUG} optimized ${FREEIMAGEPLUS_LIBRARY_RELEASE}
                      debug ${FREEIMAGEPLUS_LIBRARY_DEBUG} optimized ${FREEIMAGEPLUS_LIBRARY_RELEASE}
                      )
ELSE (MSVC)
TARGET_LINK_LIBRARIES(${_EXE_NAME} 
		                  gpucast_gl 
                      gpucast_glut
	                    gpucast_core
		                  ${FREEGLUT_LIBRARY}
		                  ${FREEIMAGE_LIBRARY}
		                  ${FREEIMAGEPLUS_LIBRARY}
		                  ${GLEW_LIBRARY}
                      boost_filesystem
                      boost_regex
                      GL GLU )
ENDIF (MSVC)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
IF (MSVC)
    ADD_CUSTOM_COMMAND ( TARGET ${_EXE_NAME} POST_BUILD 
      COMMAND robocopy \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\" *.dll /R:0 /W:0 /NP 
            & robocopy \"${GLOBAL_EXT_DIR}/lib/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\" *.dll /R:0 /W:0 /NP \n if %ERRORLEVEL% LEQ 7 (exit /b 0) else (exit /b 1)
    )
ENDIF (MSVC)