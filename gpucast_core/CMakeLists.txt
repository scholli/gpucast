###############################################################################
# determine source and header files
###############################################################################

FILE ( GLOB GPUCAST_CORE_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/core/*.hpp )
FILE ( GLOB GPUCAST_CORE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp )

FILE ( GLOB GPUCAST_CORE_IMPORT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/core/import/*.hpp )
FILE ( GLOB GPUCAST_CORE_IMPORT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/import/*.cpp )

FILE ( GLOB_RECURSE GPUCAST_CORE_GLSL RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} glsl/trimmed_surface/* )
FILE ( GLOB_RECURSE GPUCAST_CORE_GLSL_MATH RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} glsl/math/* )

SOURCE_GROUP("Header Files" FILES ${GPUCAST_CORE_INC})
SOURCE_GROUP("Source Files" FILES ${GPUCAST_CORE_SRC})

SOURCE_GROUP("Header Files\\import" FILES ${GPUCAST_CORE_IMPORT_INC})
SOURCE_GROUP("Source Files\\import" FILES ${GPUCAST_CORE_IMPORT_SRC})

SOURCE_GROUP("Header Files\\glsl" FILES ${GPUCAST_CORE_GLSL})
SOURCE_GROUP("Source Files\\glsl" FILES ${GPUCAST_CORE_GLSL_MATH})

LINK_DIRECTORIES(${LIB_PATHS} ${GLEW_LIBRARY_DIRS} ${BOOST_LIBRARY_DIRS} ${QHULL_LIBRARY_DIRS})

ADD_LIBRARY( gpucast_core SHARED
           ${GPUCAST_CORE_INC}
           ${GPUCAST_CORE_SRC}
           ${GPUCAST_CORE_IMPORT_INC}
           ${GPUCAST_CORE_IMPORT_SRC}
)

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${GLEW_INCLUDE_DIRS} 
                     ${QHULL_INCLUDE_DIRS} 
                     ${CMAKE_SOURCE_DIR}/gpucast_math/include 
                     ${CMAKE_SOURCE_DIR}/gpucast_gl/include
)

###############################################################################
# set preprocessor configuration
###############################################################################
SET (GPUCAST_CORE_BUILD_FLAGS ${BUILD_FLAGS})

IF (MSVC)
  SET(GPUCAST_CORE_BUILD_FLAGS "${GPUCAST_CORE_BUILD_FLAGS} -D GPUCAST_CORE_EXPORT")
ENDIF (MSVC)

SET_TARGET_PROPERTIES( gpucast_core PROPERTIES COMPILE_FLAGS ${GPUCAST_CORE_BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################

ADD_DEPENDENCIES(gpucast_core gpucast_gl gpucast_math)

LIST(APPEND LIBRARIES gpucast_gl ${GLEW_LIBRARIES} ${QHULL_LIBRARIES})

IF (MSVC)
    LIST(APPEND LIBRARIES OpenGL32.lib)
ELSEIF (UNIX)
ENDIF (MSVC)

LIST(LENGTH LIBRARIES NUMBER_OF_LIBRARIES)

IF ( ${NUMBER_OF_LIBRARIES} GREATER 0 )
    TARGET_LINK_LIBRARIES( gpucast_core debug ${LIBRARIES} optimized ${LIBRARIES} )
ENDIF ( ${NUMBER_OF_LIBRARIES} GREATER 0 )
