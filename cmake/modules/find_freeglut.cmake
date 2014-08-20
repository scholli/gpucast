##############################################################################
# search paths
##############################################################################
SET(FREEGLUT_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/freeglut
  ${FREEGLUT_INCLUDE_SEARCH_DIR}
  /usr/include
)

SET(FREEGLUT_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${FREEGLUT_LIBRARY_SEARCH_DIR}
  /usr/lib
  /usr/lib/x86_64-linux-gnu
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for FREEGLUT")

find_path(FREEGLUT_INCLUDE_DIR NAMES GL/freeglut.h PATHS ${FREEGLUT_INCLUDE_SEARCH_DIRS})

IF (MSVC)
	find_library(FREEGLUT_LIBRARY_RELEASE NAMES freeglut.lib PATHS ${FREEGLUT_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES release)
	find_library(FREEGLUT_LIBRARY_DEBUG NAMES freeglut.lib freeglutd.lib PATHS ${FREEGLUT_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES debug)
ELSEIF (UNIX)
	find_library(FREEGLUT_LIBRARY NAMES libglut.so PATHS ${FREEGLUT_LIBRARY_SEARCH_DIRS})
ENDIF (MSVC)
