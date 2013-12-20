##############################################################################
# search paths
##############################################################################
SET(FREEIMAGE_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/inc/freeimage
  ${FREEIMAGE_INCLUDE_SEARCH_DIR}
)

SET(FREEIMAGE_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${FREEIMAGE_LIBRARY_SEARCH_DIR}
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for FREEIMAGE")

IF (MSVC)
	find_dependency(FREEIMAGE ${FREEIMAGE_INCLUDE_SEARCH_DIRS} ${FREEIMAGE_LIBRARY_SEARCH_DIRS} FreeImage.h FreeImage.lib )
	LIST(APPEND FREEIMAGE_LIBRARIES FreeImagePlus.lib)
ELSEIF (UNIX)
	find_dependency(FREEIMAGE ${FREEIMAGE_INCLUDE_SEARCH_DIRS} ${FREEIMAGE_LIBRARY_SEARCH_DIRS} FreeImage.h FreeImage.a )
ENDIF (MSVC)

verify_dependency(FREEIMAGE)
