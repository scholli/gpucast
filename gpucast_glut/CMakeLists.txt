###############################################################################
# determine source and header files
###############################################################################
FILE ( GLOB_RECURSE GPUCAST_GLUT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/*.hpp )
FILE ( GLOB_RECURSE GPUCAST_GLUT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp )

SOURCE_GROUP("Header Files" FILES ${GPUCAST_GLUT_INC})
SOURCE_GROUP("Source Files" FILES ${GPUCAST_GLUT_SRC})

LINK_DIRECTORIES(${LIB_PATHS} ${GLEW_LIBRARY_DIRS} ${BOOST_LIBRARY_DIRS} ${FREEGLUT_LIBRARY_DIRS})

ADD_LIBRARY( gpucast_glut SHARED
           ${GPUCAST_GLUT_INC}
           ${GPUCAST_GLUT_SRC}
)

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${GLEW_INCLUDE_DIR}
                     include 
                     ../gpucast_math/include
                     ../gpucast_core/include
                     ../gpucast_gl/include
                     ${FREEGLUT_INCLUDE_DIR}
                    )

###############################################################################
# set preprocessor configuration
###############################################################################
SET (GPUCAST_GLUT_BUILD_FLAGS ${BUILD_FLAGS})

IF (MSVC)
  SET(GPUCAST_GLUT_BUILD_FLAGS "${GPUCAST_GLUT_BUILD_FLAGS} -D GPUCAST_GLUT_EXPORT")
ENDIF (MSVC)

SET_TARGET_PROPERTIES( gpucast_glut PROPERTIES COMPILE_FLAGS ${GPUCAST_GLUT_BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(gpucast_glut gpucast_math gpucast_gl)

IF (MSVC)
  TARGET_LINK_LIBRARIES( gpucast_glut
                         OpenGL32.lib 
                         debug ${FREEGLUT_LIBRARY_DEBUG} optimized ${FREEGLUT_LIBRARY_RELEASE}
                         debug ${GLEW_LIBRARY_DEBUG} optimized ${GLEW_LIBRARY_RELEASE}
                       )

ELSEIF (UNIX)
  TARGET_LINK_LIBRARIES( gpucast_glut
                         ${GLEW_LIBRARY} 
                         ${FREEGLUT_LIBRARY}
                         GL GLU
                       )
ENDIF (MSVC)

###############################################################################
# install 
###############################################################################

IF (MSVC)
  install (TARGETS gpucast_glut
    CONFIGURATIONS Release
    RUNTIME DESTINATION bin/Release
    LIBRARY DESTINATION lib/Release
    ARCHIVE DESTINATION lib/Release
  )

  install (TARGETS gpucast_glut
    CONFIGURATIONS Debug
    RUNTIME DESTINATION bin/Debug
    LIBRARY DESTINATION lib/Debug
    ARCHIVE DESTINATION lib/Debug
  )
ELSEIF (UNIX)
  install (TARGETS gpucast_glut 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
ENDIF (MSVC)

# header files 
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gpucast DESTINATION include/gpucast_glut FILES_MATCHING PATTERN "*.hpp")

