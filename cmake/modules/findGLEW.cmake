##############################################################################
# search paths
##############################################################################
SET(GLEW_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/glew/include
  ${GLEW_INCLUDE_SEARCH_DIR}
  /usr/include
)

SET(GLEW_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/glew/lib
  ${GLEW_LIBRARY_SEARCH_DIR}
  /usr/lib
  /usr/lib/x86_64-linux-gnu
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for GLEW")

find_path(GLEW_INCLUDE_DIR NAMES GL/glew.h PATHS ${GLEW_INCLUDE_SEARCH_DIRS})

IF (MSVC)
	find_library(GLEW_LIBRARY_DEBUG NAMES glew32.lib glew32d.lib PATHS ${GLEW_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES debug)
  find_library(GLEW_LIBRARY_RELEASE NAMES glew32.lib PATHS ${GLEW_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES release)
ELSEIF (UNIX)
	find_library(GLEW_LIBRARY NAMES libGLEW.so PATHS ${GLEW_LIBRARY_SEARCH_DIRS})
ENDIF (MSVC)
