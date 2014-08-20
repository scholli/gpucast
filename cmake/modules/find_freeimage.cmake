##############################################################################
# search paths
##############################################################################
SET(FREEIMAGE_INCLUDE_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/include/freeimage
  ${FREEIMAGE_INCLUDE_SEARCH_DIR}
  /usr/include
)

SET(FREEIMAGE_LIBRARY_SEARCH_DIRS
  ${GLOBAL_EXT_DIR}/lib
  ${FREEIMAGE_LIBRARY_SEARCH_DIR}
  /usr/lib
  /usr/lib/x86_64-linux-gnu
)

##############################################################################
# search
##############################################################################
message(STATUS "-- checking for FREEIMAGE")

find_path(FREEIMAGE_INCLUDE_DIR NAMES FreeImage.h FreeImagePlus.h PATHS ${FREEIMAGE_INCLUDE_SEARCH_DIRS})

IF (MSVC)
	find_library(FREEIMAGE_LIBRARY_RELEASE NAMES FreeImage.lib PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES release)
	find_library(FREEIMAGEPLUS_LIBRARY_RELEASE NAMES FreeImagePlus.lib PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES release)

	find_library(FREEIMAGE_LIBRARY_DEBUG NAMES FreeImaged.lib FreeImage.lib PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES debug)
	find_library(FREEIMAGEPLUS_LIBRARY_DEBUG NAMES FreeImagePlus.lib FreeImagePlusd.lib PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS} PATH_SUFFIXES debug)
ELSEIF (UNIX)
	find_library(FREEIMAGE_LIBRARY NAMES libfreeimage.so PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS})
	find_library(FREEIMAGEPLUS_LIBRARY NAMES libfreeimageplus.so PATHS ${FREEIMAGE_LIBRARY_SEARCH_DIRS})
ENDIF (MSVC)
