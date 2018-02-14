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

FILE ( GLOB GPUCAST_GL_GLSL_SURFACE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../resources/glsl/trimmed_surface/* )
FILE ( GLOB GPUCAST_GL_GLSL_MATH RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../resources/glsl/math/* )
FILE ( GLOB GPUCAST_GL_GLSL_TRIMMING RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../resources/glsl/trimming/* )
FILE ( GLOB GPUCAST_GL_GLSL_BASE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../resources/glsl/base/* )
FILE ( GLOB GPUCAST_GL_GLSL_COMMON RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../resources/glsl/common/* )
FILE ( GLOB GPUCAST_GL_GLSL_ABUFFER RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../resources/glsl/abuffer/* )

SOURCE_GROUP("Resources\\abuffer" FILES ${GPUCAST_GL_GLSL_ABUFFER})
SOURCE_GROUP("Resources\\base" FILES ${GPUCAST_GL_GLSL_BASE})
SOURCE_GROUP("Resources\\math" FILES ${GPUCAST_GL_GLSL_MATH})
SOURCE_GROUP("Resources\\trimmed_surface" FILES ${GPUCAST_GL_GLSL_SURFACE})
SOURCE_GROUP("Resources\\trimming" FILES ${GPUCAST_GL_GLSL_TRIMMING})
SOURCE_GROUP("Resources\\common" FILES ${GPUCAST_GL_GLSL_COMMON})

SOURCE_GROUP("Source Files" FILES ${GPUCAST_GL_INC})
SOURCE_GROUP("Source Files\\glut" FILES ${GPUCAST_GL_GLUT_INC})
SOURCE_GROUP("Source Files\\graph" FILES ${GPUCAST_GL_GRAPH_INC})
SOURCE_GROUP("Source Files\\import" FILES ${GPUCAST_GL_IMPORT_INC})
SOURCE_GROUP("Source Files\\math" FILES ${GPUCAST_GL_MATH_INC})
SOURCE_GROUP("Source Files\\primitives" FILES ${GPUCAST_GL_PRIM_INC})
SOURCE_GROUP("Source Files\\test" FILES ${GPUCAST_GL_TEST_INC})
SOURCE_GROUP("Source Files\\util" FILES ${GPUCAST_GL_UTIL_INC})

SOURCE_GROUP("Source Files" FILES ${GPUCAST_GL_SRC})
SOURCE_GROUP("Source Files\\glut" FILES ${GPUCAST_GL_GLUT_SRC})
SOURCE_GROUP("Source Files\\graph" FILES ${GPUCAST_GL_GRAPH_SRC})
SOURCE_GROUP("Source Files\\import" FILES ${GPUCAST_GL_IMPORT_SRC})
SOURCE_GROUP("Source Files\\math" FILES ${GPUCAST_GL_MATH_SRC})
SOURCE_GROUP("Source Files\\primitives" FILES ${GPUCAST_GL_PRIM_SRC})
SOURCE_GROUP("Source Files\\test" FILES ${GPUCAST_GL_TEST_SRC})
SOURCE_GROUP("Source Files\\util" FILES ${GPUCAST_GL_UTIL_SRC})

LINK_DIRECTORIES(${LIB_PATHS} ${GLEW_LIBRARY_DIRS} ${QHULL_LIBRARY_DIRS} ${FREEIMAGE_LIBRARY_DIRS})

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
           ${GPUCAST_GL_GLSL_ABUFFER}
           ${GPUCAST_GL_GLSL_MATH}
           ${GPUCAST_GL_GLSL_SURFACE}
           ${GPUCAST_GL_GLSL_TRIMMING}
           ${GPUCAST_GL_GLSL_BASE}
           ${GPUCAST_GL_GLSL_COMMON}
)

set_target_properties(gpucast_gl PROPERTIES DEBUG_POSTFIX "d")

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

TARGET_LINK_LIBRARIES( gpucast_gl
                       gpucast_core
                       debug ${FREEIMAGE_LIBRARY_DEBUG} optimized ${FREEIMAGE_LIBRARY}
                       debug ${FREEIMAGE_PLUS_LIBRARY_DEBUG} optimized ${FREEIMAGE_PLUS_LIBRARY}
                       general ${GLEW_LIBRARY}     
                       debug ${Boost_SYSTEM_LIBRARY_DEBUG} optimized ${Boost_SYSTEM_LIBRARY_RELEASE}
                       debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
                       debug ${Boost_REGEX_LIBRARY_DEBUG} optimized ${Boost_REGEX_LIBRARY_RELEASE}
                       debug ${Boost_LOG_LIBRARY_DEBUG} optimized ${Boost_LOG_LIBRARY_RELEASE}
                       debug ${Boost_LOG_SETUP_LIBRARY_DEBUG} optimized ${Boost_LOG_SETUP_LIBRARY_RELEASE}
                     )

IF (MSVC)
  TARGET_LINK_LIBRARIES (gpucast_gl OpenGL32.lib)
ENDIF (MSVC)                     

###############################################################################
# install 
###############################################################################

IF (MSVC)
  install (TARGETS gpucast_gl    
    CONFIGURATIONS Release
    RUNTIME DESTINATION bin/Release
    LIBRARY DESTINATION lib/Release
    ARCHIVE DESTINATION lib/Release
  )

  install (TARGETS gpucast_gl
    CONFIGURATIONS Debug
    RUNTIME DESTINATION bin/Debug
    LIBRARY DESTINATION lib/Debug
    ARCHIVE DESTINATION lib/Debug
  )
ELSEIF (UNIX)
  install (TARGETS gpucast_gl
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
ENDIF (MSVC)

# header files 
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gpucast DESTINATION include FILES_MATCHING PATTERN "*.hpp")
