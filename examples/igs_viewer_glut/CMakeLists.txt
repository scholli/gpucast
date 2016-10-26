###############################################################################
# set sources
###############################################################################
FILE(GLOB EXAMPLE_IGS_VIEWER_GLUT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp *.cpp *.hpp include/*.hpp)

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
    ${EXAMPLE_IGS_VIEWER_GLUT_SRC}
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
            general ${GLUT_glut_LIBRARY}
            general ${GLEW_LIBRARY}     
            debug ${QHULL_LIBRARY_DEBUG}          optimized ${QHULL_LIBRARY_RELEASE}        
            debug ${FREEIMAGE_LIBRARY_DEBUG}      optimized ${FREEIMAGE_LIBRARY}    
            debug ${FREEIMAGE_PLUS_LIBRARY_DEBUG} optimized ${FREEIMAGE_PLUS_LIBRARY} 
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
		      ${QHULL_LIBRARY}
          boost_filesystem boost_regex
          GL GLU
          )
ENDIF (MSVC)

###############################################################################
# copy runtime libraries as a post-build process
###############################################################################
copy_external_runtime_libraries(freeglut glew freeimage)
apply_debug_working_directory()