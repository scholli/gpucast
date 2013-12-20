###############################################################################
# set sources
###############################################################################
file(GLOB EXAMPLE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp include/*.hpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_SRC}
)

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

LIST(APPEND LIBRARIES ${FREEGLUT_LIBRARIES} gpucast_gl gpucast_core)
LIST(LENGTH LIBRARIES NUMBER_OF_LIBRARIES)

IF ( ${NUMBER_OF_LIBRARIES} GREATER 0 )
    TARGET_LINK_LIBRARIES(${_EXE_NAME} debug ${LIBRARIES} optimized ${LIBRARIES} )
ENDIF ( ${NUMBER_OF_LIBRARIES} GREATER 0 )

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
IF (MSVC)
    copy_runtime_dependencies ( ${_EXE_NAME} \"${GLOBAL_EXT_DIR}/bin/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    copy_runtime_dependencies ( ${_EXE_NAME} \"${LIBRARY_OUTPUT_PATH}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    FOREACH (_LIB_DIR ${LIB_PATHS})
        copy_runtime_dependencies ( ${_EXE_NAME} \"${_LIB_DIR}/$(Configuration)/\" \"${EXECUTABLE_OUTPUT_PATH}/$(Configuration)/\")
    ENDFOREACH (_LIB_DIR ${LIB_PATHS})
ENDIF (MSVC)