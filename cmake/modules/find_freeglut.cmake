##############################################################################
# search paths
##############################################################################
SET(FREEGLUT_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/freeglut
  ${FREEGLUT_INCLUDE_SEARCH_DIR}
)

SET(FREEGLUT_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${FREEGLUT_LIBRARY_SEARCH_DIR}
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for FREEGLUT")

IF (MSVC)
	find_dependency(FREEGLUT ${FREEGLUT_INCLUDE_SEARCH_DIRS} ${FREEGLUT_LIBRARY_SEARCH_DIRS} GL/freeglut.h freeglut.lib )
ELSEIF (UNIX)
	find_dependency(FREEGLUT ${FREEGLUT_INCLUDE_SEARCH_DIRS} ${FREEGLUT_LIBRARY_SEARCH_DIRS} GL/freeglut.h freeglut.a )
ENDIF (MSVC)

verify_dependency(FREEGLUT)
