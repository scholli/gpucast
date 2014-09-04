###############################################################################
# determine source and header files
###############################################################################
FILE ( GLOB GPUCAST_GL_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/*.hpp )
FILE ( GLOB GPUCAST_GL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp )

FILE ( GLOB GPUCAST_GL_TEST_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/test/*.hpp )
FILE ( GLOB GPUCAST_GL_TEST_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/test/*.cpp )

FILE ( GLOB GPUCAST_GL_GRAPH_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/graph/*.hpp )
FILE ( GLOB GPUCAST_GL_GRAPH_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/graph/*.cpp )

FILE ( GLOB GPUCAST_GL_IMPORT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/import/*.hpp )
FILE ( GLOB GPUCAST_GL_IMPORT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/import/*.cpp )

FILE ( GLOB GPUCAST_GL_PRIM_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/primitives/*.hpp )
FILE ( GLOB GPUCAST_GL_PRIM_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/primitives/*.cpp )

FILE ( GLOB GPUCAST_GL_MATH_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/math/*.hpp )

FILE ( GLOB GPUCAST_GL_UTIL_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/util/*.hpp )
FILE ( GLOB GPUCAST_GL_UTIL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/util/*.cpp )

FILE ( GLOB GPUCAST_GL_GLUT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/glut/*.cpp src/glut/display/*.cpp )
FILE ( GLOB GPUCAST_GL_GLUT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/gl/glut/*.hpp include/gpucast/gl/glut/display/*.hpp)

SOURCE_GROUP("Header Files" FILES ${GPUCAST_GL_INC})
SOURCE_GROUP("Header Files\\glut" FILES ${GPUCAST_GL_GLUT_INC})
SOURCE_GROUP("Header Files\\graph" FILES ${GPUCAST_GL_GRAPH_INC})
SOURCE_GROUP("Header Files\\import" FILES ${GPUCAST_GL_IMPORT_INC})
SOURCE_GROUP("Header Files\\math" FILES ${GPUCAST_GL_MATH_INC})
SOURCE_GROUP("Header Files\\primitives" FILES ${GPUCAST_GL_PRIM_INC})
SOURCE_GROUP("Header Files\\test" FILES ${GPUCAST_GL_TEST_INC})
SOURCE_GROUP("Header Files\\util" FILES ${GPUCAST_GL_UTIL_INC})

SOURCE_GROUP("Source Files" FILES ${GPUCAST_GL_SRC})
SOURCE_GROUP("Source Files\\glut" FILES ${GPUCAST_GL_GLUT_SRC})
SOURCE_GROUP("Source Files\\graph" FILES ${GPUCAST_GL_GRAPH_SRC})
SOURCE_GROUP("Source Files\\import" FILES ${GPUCAST_GL_IMPORT_SRC})
SOURCE_GROUP("Source Files\\math" FILES ${GPUCAST_GL_MATH_SRC})
SOURCE_GROUP("Source Files\\primitives" FILES ${GPUCAST_GL_PRIM_SRC})
SOURCE_GROUP("Source Files\\test" FILES ${GPUCAST_GL_TEST_SRC})
SOURCE_GROUP("Source Files\\util" FILES ${GPUCAST_GL_UTIL_SRC})

LINK_DIRECTORIES(${LIB_PATHS} ${GLEW_LIBRARY_DIRS} ${BOOST_LIBRARY_DIRS} ${QHULL_LIBRARY_DIRS} ${FREEIMAGE_LIBRARY_DIRS})

ADD_LIBRARY( gpucast_gl SHARED
           ${GPUCAST_GL_INC}
           ${GPUCAST_GL_GLUT_INC}
    	     ${GPUCAST_GL_GRAPH_INC}
           ${GPUCAST_GL_IMPORT_INC}
           ${GPUCAST_GL_PRIM_INC}
           ${GPUCAST_GL_MATH_INC}
           ${GPUCAST_GL_TEST_INC}
           ${GPUCAST_GL_UTIL_INC}
           ${GPUCAST_GL_SRC}
           ${GPUCAST_GL_GRAPH_SRC}
           ${GPUCAST_GL_IMPORT_SRC}
           ${GPUCAST_GL_GLUT_SRC}
           ${GPUCAST_GL_MATH_SRC}
           ${GPUCAST_GL_PRIM_SRC}
           ${GPUCAST_GL_UTIL_SRC}          
           ${GPUCAST_GL_TEST_SRC} 
)

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${GLEW_INCLUDE_DIR}
                     ../gpucast_core/include
                     ${FREEIMAGE_INCLUDE_DIR}
                     include 
                     ../gpucast_math/include )

###############################################################################
# set preprocessor configuration
###############################################################################
SET (GPUCAST_GL_BUILD_FLAGS ${BUILD_FLAGS})

IF (MSVC)
  SET(GPUCAST_GL_BUILD_FLAGS "${GPUCAST_GL_BUILD_FLAGS} -D GPUCAST_GL_EXPORT")
ENDIF (MSVC)

SET_TARGET_PROPERTIES( gpucast_gl PROPERTIES COMPILE_FLAGS ${GPUCAST_GL_BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(gpucast_gl gpucast_math)

IF (MSVC)
  TARGET_LINK_LIBRARIES( gpucast_gl
                         OpenGL32.lib 
                         debug ${FREEIMAGE_LIBRARY_DEBUG} optimized ${FREEIMAGE_LIBRARY_RELEASE}
                         debug ${FREEIMAGEPLUS_LIBRARY_DEBUG} optimized ${FREEIMAGEPLUS_LIBRARY_RELEASE}
                         debug ${GLEW_LIBRARY_DEBUG} optimized ${GLEW_LIBRARY_RELEASE}
                       )

ELSE (MSVC)
  TARGET_LINK_LIBRARIES( gpucast_gl
                         ${GLEW_LIBRARY} 
                         ${FREEIMAGE_LIBRARY}
                         ${FREEIMAGEPLUS_LIBRARY}
                         GL GLU
                       )
ENDIF (MSVC)

