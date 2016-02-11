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
                     ${QHULL_INCLUDE_DIR} 
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

IF (MSVC)
    TARGET_LINK_LIBRARIES( gpucast_core 
                           debug ${QHULL_LIBRARY_DEBUG} optimized ${QHULL_LIBRARY_RELEASE}
                           debug ${Boost_SYSTEM_LIBRARY_DEBUG} optimized ${Boost_SYSTEM_LIBRARY_RELEASE}
                           debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
                           OpenGL32.lib )
ELSEIF (UNIX)
    TARGET_LINK_LIBRARIES( gpucast_core 
                           ${QHULL_LIBRARY}
                           GL )
ENDIF (MSVC)

###############################################################################
# install 
###############################################################################

IF (MSVC)
  install (TARGETS gpucast_core
    CONFIGURATIONS Release
    RUNTIME DESTINATION bin/Release
    LIBRARY DESTINATION lib/Release
    ARCHIVE DESTINATION lib/Release
  )

  install (TARGETS gpucast_core
    CONFIGURATIONS Debug
    RUNTIME DESTINATION bin/Debug
    LIBRARY DESTINATION lib/Debug
    ARCHIVE DESTINATION lib/Debug
  )
ELSEIF (UNIX)
  install (TARGETS gpucast_core 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
ENDIF (MSVC)

# header files 
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gpucast DESTINATION include FILES_MATCHING PATTERN "*.hpp")

# shader and material resources
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../resources/glsl DESTINATION resources
        FILES_MATCHING 
        PATTERN "*.frag" 
        PATTERN "*.vert" 
        PATTERN "*.glsl")
