##############################################################################
# search paths
##############################################################################
SET(GLFW_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/GLFW
  ${GLFW_INCLUDE_SEARCH_DIR}
)

SET(GLFW_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${GLFW_LIBRARY_SEARCH_DIR}
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for GLFW")

IF (MSVC)
	find_dependency(GLFW ${GLFW_INCLUDE_SEARCH_DIRS} ${GLFW_LIBRARY_SEARCH_DIRS} GLFW/glfw3.h glfw3.lib )
ELSEIF (UNIX)
	find_dependency(GLFW ${GLFW_INCLUDE_SEARCH_DIRS} ${GLFW_LIBRARY_SEARCH_DIRS} GLFW/glfw3.h glfw3.a )
ENDIF (MSVC)

verify_dependency(GLFW)
