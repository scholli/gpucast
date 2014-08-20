##############################################################################
# search paths
##############################################################################
SET(UNITTEST_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/unittest
  ${UNITTEST_INCLUDE_SEARCH_DIR}
  /usr/include
)

SET(UNITTEST_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${UNITTEST_LIBRARY_SEARCH_DIR}
  /usr/lib
  /usr/lib/x86_64-linux-gnu
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for UNITTEST")

find_path(UNITTEST_INCLUDE_DIR NAMES unittest++/UnitTest++.h PATHS ${UNITTEST_INCLUDE_SEARCH_DIRS})

IF (MSVC)
	find_library(UNITTEST_LIBRARY NAMES UnitTest++.lib PATHS ${UNITTEST_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES release debug)
ELSEIF (UNIX)
	find_library(UNITTEST_LIBRARY NAMES libUnitTest++.a PATHS ${UNITTEST_LIBRARY_SEARCH_DIRS})
ENDIF (MSVC)
