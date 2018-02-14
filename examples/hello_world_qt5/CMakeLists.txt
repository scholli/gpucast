if(COMMAND cmake_policy)
    cmake_policy(SET CMP0020 OLD)
endif()


list(APPEND CMAKE_PREFIX_PATH "${GLOBAL_EXT_DIR}/Qt5/lib/cmake")

find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)

###############################################################################
# set sources
###############################################################################
file(GLOB HELLO_WORLD_QT5_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp include/*.hpp)

GET_FILENAME_COMPONENT(_EXE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_EXE_NAME example_${_EXE_NAME}.out)
PROJECT(${_EXE_NAME})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                     ${CMAKE_SOURCE_DIR}/gpucast_core/include
                     ${Qt5Core_INCLUDE_DIRS}
                     ${Qt5Gui_INCLUDE_DIRS}
                     ${Qt5Widgets_INCLUDE_DIRS}
                     ${Qt5OpenGL_INCLUDE_DIRS}
                     ${GLEW_INCLUDE_DIRS}
)

SET(LIBRARY_DIRS ${LIB_PATHS} 
                 ${GLEW_LIBRARY_DIRS} 
                 ${BOOST_LIBRARY_DIRS} 
		             


${QT_LIBRARY_DIRS}
)

LINK_DIRECTORIES (${LIBRARY_DIRS})

ADD_EXECUTABLE( ${_EXE_NAME}
    ${HELLO_WORLD_QT5_SRC}
)

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(${_EXE_NAME} gpucast_math gpucast_gl gpucast_core)

TARGET_LINK_LIBRARIES(${_EXE_NAME} 
                      gpucast_gl 
                      gpucast_core 
                      ${Qt5OpenGL_LIBRARIES}
                      ${Qt5Core_LIBRARIES}
                      ${Qt5Widgets_LIBRARIES}
                      ${Qt5Gui_LIBRARIES}
                      ${QT_QTMAIN_LIBRARY}
)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
copy_external_runtime_libraries(qt5 glew freeimage)
apply_debug_working_directory()