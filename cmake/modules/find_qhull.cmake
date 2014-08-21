##############################################################################
# search paths
##############################################################################
SET(QHULL_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/qhull
  ${QHULL_INCLUDE_SEARCH_DIR}
  /usr/include
)

SET(QHULL_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${QHULL_LIBRARY_SEARCH_DIR}
  /usr/lib
  /usr/lib/x86_64-linux-gnu
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for QHULL")

find_path(QHULL_INCLUDE_DIR NAMES libqhull/libqhull.h PATHS ${QHULL_INCLUDE_SEARCH_DIRS})

IF (MSVC)
	find_library(QHULL_LIBRARY_DEBUG NAMES qhullstatic.lib PATHS ${QHULL_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES debug)
  find_library(QHULL_LIBRARY_RELEASE NAMES qhullstatic.lib PATHS ${QHULL_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES release)
ELSEIF (UNIX)
	find_library(QHULL_LIBRARY NAMES libqhull.so PATHS ${QHULL_LIBRARY_SEARCH_DIRS})
ENDIF (MSVC)
