##############################################################################
# search paths
##############################################################################
SET(UNITTEST_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/inc/unittest
  ${UNITTEST_INCLUDE_SEARCH_DIR}
)

SET(UNITTEST_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${UNITTEST_LIBRARY_SEARCH_DIR}
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for UNITTEST")

IF (MSVC)
	find_dependency(UNITTEST ${UNITTEST_INCLUDE_SEARCH_DIRS} ${UNITTEST_LIBRARY_SEARCH_DIRS} UnitTest++.h UnitTest++.lib )
ELSEIF (UNIX)
	find_dependency(UNITTEST ${UNITTEST_INCLUDE_SEARCH_DIRS} ${UNITTEST_LIBRARY_SEARCH_DIRS} UnitTest++.h UnitTest++.a )
ENDIF (MSVC)

verify_dependency(UNITTEST)
