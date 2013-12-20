##############################################################################
# search paths
##############################################################################
SET(GLEW_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/inc/glew
  ${GLEW_INCLUDE_SEARCH_DIR}
)

SET(GLEW_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${GLEW_LIBRARY_SEARCH_DIR}
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for GLEW")

IF (MSVC)
	find_dependency(GLEW ${GLEW_INCLUDE_SEARCH_DIRS} ${GLEW_LIBRARY_SEARCH_DIRS} GL/glew.h glew32.lib )
ELSEIF (UNIX)
	find_dependency(GLEW ${GLEW_INCLUDE_SEARCH_DIRS} ${GLEW_LIBRARY_SEARCH_DIRS} GL/glew.h glew32.a )
ENDIF (MSVC)

verify_dependency(GLEW)
