##############################################################################
# search paths
##############################################################################
SET(QHULL_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/qhull
  ${QHULL_INCLUDE_SEARCH_DIR}
)

SET(QHULL_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${QHULL_LIBRARY_SEARCH_DIR}
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for QHULL")

IF (MSVC)
	find_dependency(QHULL ${QHULL_INCLUDE_SEARCH_DIRS} ${QHULL_LIBRARY_SEARCH_DIRS} libqhull/qhull_a.h qhullstatic.lib )
  LIST(APPEND QHULL_LIBRARIES qhullstatic_p.lib)
ELSEIF (UNIX)
	find_dependency(QHULL ${QHULL_INCLUDE_SEARCH_DIRS} ${QHULL_LIBRARY_SEARCH_DIRS} libqhull/qhull_a.h qhull.a )
ENDIF (MSVC)

verify_dependency(QHULL)
