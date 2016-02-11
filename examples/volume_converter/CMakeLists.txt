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
                        debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
                        debug ${Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG} optimized ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}
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
  FOREACH(_LIB ${GPUCAST_RUNTIME_LIBRARIES})
    get_filename_component(_FILE ${_LIB} NAME)
    get_filename_component(_PATH ${_LIB} DIRECTORY)
    SET(COPY_DLL_COMMAND_STRING ${COPY_DLL_COMMAND_STRING} robocopy \"${_PATH}\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\" ${_FILE} /R:0 /W:0 /NP &)
  ENDFOREACH()

  SET(COPY_DLL_COMMAND_STRING ${COPY_DLL_COMMAND_STRING} robocopy \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\" *.dll /R:0 /W:0 /NP &)
  ADD_CUSTOM_COMMAND ( TARGET ${_EXE_NAME} POST_BUILD COMMAND ${COPY_DLL_COMMAND_STRING} \n if %ERRORLEVEL% LEQ 7 (exit /b 0) else (exit /b 1))
ENDIF (MSVC)

ENDIF (${ENABLE_GPUCAST_VOLUME_MODULE})