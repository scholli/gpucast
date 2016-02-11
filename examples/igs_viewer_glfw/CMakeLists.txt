include(findGLFW)

###############################################################################
# set sources
###############################################################################
FILE(GLOB EXAMPLE_IGS_VIEWER_GLFW_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.hpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
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
    ${EXAMPLE_IGS_VIEWER_GLFW_SRC}
)

SET_TARGET_PROPERTIES( ${_EXE_NAME} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

IF (MSVC)
  TARGET_LINK_LIBRARIES(${_EXE_NAME}  
		        gpucast_gl 
		        gpucast_core   
            debug ${GLFW_LIBRARY_DEBUG}           optimized ${GLFW_LIBRARY_RELEASE}         
            debug ${GLEW_LIBRARY_DEBUG}           optimized ${GLEW_LIBRARY_RELEASE}         
            debug ${QHULL_LIBRARY_DEBUG}          optimized ${QHULL_LIBRARY_RELEASE}        
            debug ${FREEIMAGE_LIBRARY_DEBUG}      optimized ${FREEIMAGE_LIBRARY}    
            debug ${FREEIMAGE_PLUS_LIBRARY_DEBUG} optimized ${FREEIMAGE_PLUS_LIBRARY}
            general ${GLUT_glut_LIBRARY}
  )
ELSEIF (UNIX)
  TARGET_LINK_LIBRARIES(${_EXE_NAME}  
		        gpucast_gl 
		        gpucast_core 
            ${GLFW_LIBRARY}
            ${GLEW_LIBRARY}
            ${QHULL_LIBRARY}
            ${FREEIMAGE_LIBRARY}
            ${FREEIMAGEPLUS_LIBRARY}
            ${FREEGLUT_LIBRARY}
            boost_thread boost_filesystem boost_regex
            GL GLU X11 Xi Xxf86vm Xrandr pthread
  )
ENDIF (MSVC)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
IF (MSVC)
  FOREACH(_LIB ${GPUCAST_RUNTIME_LIBRARIES})
    get_filename_component(_FILE ${_LIB} NAME)
    get_filename_component(_PATH ${_LIB} DIRECTORY)
    SET(COPY_DLL_COMMAND_STRING ${COPY_DLL_COMMAND_STRING} robocopy \"${_PATH}\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\" ${_FILE} /R:0 /W:0 /NP &)
  ENDFOREACH()

  SET(COPY_DLL_COMMAND_STRING ${COPY_DLL_COMMAND_STRING} robocopy \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\" *.dll /R:0 /W:0 /NP &)
  ADD_CUSTOM_COMMAND ( TARGET ${_EXE_NAME} POST_BUILD COMMAND ${COPY_DLL_COMMAND_STRING} \n if %ERRORLEVEL% LEQ 7 (exit /b 0) else (exit /b 1))
ENDIF (MSVC)
