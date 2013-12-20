###############################################################################
# generate QT meta code
###############################################################################
include(find_qt4)

FILE(GLOB HEADER_FILES_IGS_VIEWER_QT_GUI RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/*.hpp)

add_moc_command(SOURCE_FILES_IGS_VIEWER_QT_GUI ${HEADER_FILES_IGS_VIEWER_QT_GUI})

###############################################################################
# set sources
###############################################################################
file(GLOB EXAMPLE_IGS_VIEWER_QT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp include/*.hpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${FREEGLUT_INCLUDE_DIRS}
                     ${GLEW_INCLUDE_DIRS}
                     ${QT_INCLUDE_DIRS}
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
)

SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${BOOST_LIBRARY_DIRS} 
                 ${QT_LIBRARY_DIRS}
                 ${QHULL_LIBRARY_DIRS} 
                 ${FREEIMAGE_LIBRARY_DIRS}
                 ${FREEGLUT_LIBRARY_DIRS}
)

ADD_EXECUTABLE( ${_EXE_NAME}
    ${EXAMPLE_IGS_VIEWER_QT_SRC}
    ${SOURCE_FILES_IGS_VIEWER_QT_GUI}
)

SET_TARGET_PROPERTIES( ${_EXE_NAME} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

LIST(LENGTH LIBRARIES NUMBER_OF_LIBRARIES)

IF ( ${NUMBER_OF_LIBRARIES} GREATER 0 )
    TARGET_LINK_LIBRARIES(${_EXE_NAME} general ${LIBRARIES} gpucast_gl gpucast_core debug ${QT_LIBRARIES_DEBUG} optimized ${QT_LIBRARIES_RELEASE} )
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