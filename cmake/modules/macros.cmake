##############################################################################
# retrieve list of subdirectories
##############################################################################
MACRO(get_subdirectories subdirectories current_directory)
  FILE(GLOB directory_content RELATIVE ${current_directory} *)
  SET(dirlist "")
  FOREACH(child ${directory_content}) 
    IF(IS_DIRECTORY ${current_directory}/${child})
        SET(dirlist ${dirlist} ${child})
    ENDIF()
  ENDFOREACH()
  SET(${subdirectories} ${dirlist})
ENDMACRO(get_subdirectories)

##############################################################################
# copy runtime libraries as a post-build process
##############################################################################
# copy_runtime_dependencies [target_name] [path_to_runtime_libraries] [executable_path]
MACRO(COPY_RUNTIME_DEPENDENCIES _TARGET_NAME _RUNTIME_LIBRARY_PATH _TARGET_PATH)

	SET ( _COPY_COMMAND "robocopy" )

	IF (WIN32)

    MAKE_DIRECTORY(${_TARGET_PATH})

    IF ("${_RUNTIME_LIBARY_PATH}" STREQUAL "")
      SET(_RUNTIME_LIBARY_PATH .)
    ENDIF ("${_RUNTIME_LIBARY_PATH}" STREQUAL "")

		SET(_POST_PROCESS_COMMAND ${_COPY_COMMAND} "${_RUNTIME_LIBARY_PATH}" "${_TARGET_PATH}" *.dll /R:0 /W:0)

		ADD_CUSTOM_COMMAND ( TARGET ${_TARGET_NAME} POST_BUILD COMMAND ${_POST_PROCESS_COMMAND})

	ENDIF(WIN32)
	
ENDMACRO(COPY_RUNTIME_DEPENDENCIES)


##############################################################################
# feedback to provide user-defined paths to search for python
##############################################################################
MACRO (request_search_directories dependency_name)
    
    IF ( NOT ${dependency_name}_INCLUDE_DIRS AND NOT ${dependency_name}_LIBRARY_DIRS )
        SET(${dependency_name}_INCLUDE_SEARCH_DIR "Please provide ${dependency_name} include path." CACHE PATH "path to ${dependency_name} headers.")
        SET(${dependency_name}_LIBRARY_SEARCH_DIR "Please provide ${dependency_name} library path." CACHE PATH "path to ${dependency_name} libraries.")
        MESSAGE(FATAL_ERROR "find_${dependency_name}.cmake: unable to find ${dependency_name}.")
    ENDIF ( NOT ${dependency_name}_INCLUDE_DIRS AND NOT ${dependency_name}_LIBRARY_DIRS )

    IF ( NOT ${dependency_name}_INCLUDE_DIRS )
        SET(${dependency_name}_INCLUDE_SEARCH_DIR "Please provide ${dependency_name} include path." CACHE PATH "path to ${dependency_name} headers.")
        MESSAGE(FATAL_ERROR "find_${dependency_name}.cmake: unable to find ${dependency_name} headers.")
    ELSE ( NOT ${dependency_name}_INCLUDE_DIRS )
        UNSET(${dependency_name}_INCLUDE_SEARCH_DIR CACHE)
    ENDIF ( NOT ${dependency_name}_INCLUDE_DIRS )

    IF ( NOT ${dependency_name}_LIBRARY_DIRS )
        SET(${dependency_name}_LIBRARY_SEARCH_DIR "Please provide ${dependency_name} library path." CACHE PATH "path to ${dependency_name} libraries.")
        MESSAGE(FATAL_ERROR "find_${dependency_name}.cmake: unable to find ${dependency_name} libraries.")
    ELSE ( NOT ${dependency_name}_LIBRARY_DIRS )
        UNSET(${dependency_name}_LIBRARY_SEARCH_DIR CACHE)
    ENDIF ( NOT ${dependency_name}_LIBRARY_DIRS ) 

ENDMACRO (request_search_directories dependency_name)

##############################################################################
# verify
##############################################################################
MACRO (verify_dependency dependency_name)
    
IF ( NOT ${dependency_name}_INCLUDE_DIRS OR NOT ${dependency_name}_LIBRARY_DIRS )
    request_search_directories(${dependency_name})
ELSE ( NOT ${dependency_name}_INCLUDE_DIRS OR NOT ${dependency_name}_LIBRARY_DIRS ) 
    UNSET(${dependency_name}_INCLUDE_SEARCH_DIR CACHE)
    UNSET(${dependency_name}_LIBRARY_SEARCH_DIR CACHE)
    MESSAGE(STATUS "--  found matching ${dependency_name} version")
ENDIF ( NOT ${dependency_name}_INCLUDE_DIRS OR NOT ${dependency_name}_LIBRARY_DIRS )

ENDMACRO (verify_dependency dependency_name)

##############################################################################
# Copies output of target to file
##############################################################################
MACRO ( post_build_install_target cmake_target out_file )

  GET_PROPERTY(_SOURCE_PATH TARGET ${cmake_target} PROPERTY LOCATION)
  FILE ( TO_NATIVE_PATH ${_SOURCE_PATH} _SOURCE_PATH )

  GET_FILENAME_COMPONENT(_TARGET_DIRECTORY ${out_file} PATH)

  FILE ( TO_NATIVE_PATH ${_TARGET_DIRECTORY} _TARGET_DIRECTORY )
  FILE ( TO_NATIVE_PATH ${out_file} _TARGET_PATH )

  IF (WIN32)
    ADD_CUSTOM_COMMAND(TARGET ${cmake_target}
                       POST_BUILD
                       COMMAND IF exist ${_TARGET_DIRECTORY} ( copy /Y ${_SOURCE_PATH} ${_TARGET_PATH}) ELSE ( mkdir ${_TARGET_DIRECTORY} &&  copy /Y ${_SOURCE_PATH} ${_TARGET_PATH})
                       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  ELSEIF (UNIX)
    ADD_CUSTOM_COMMAND(TARGET ${cmake_target}
                       POST_BUILD
                       COMMAND mkdir -p ${_TARGET_DIRECTORY}
                       COMMAND cp ${_SOURCE_PATH} ${_TARGET_PATH}
                       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  ENDIF (WIN32)

ENDMACRO (post_build_install_target cmake_target out_file)

##############################################################################
# print dependency
##############################################################################
MACRO (print_dependency dependency_name)
	MESSAGE(STATUS " ${dependency_name}:" )
	MESSAGE(STATUS "   library: ${${dependency_name}_LIBRARIES}" )
	MESSAGE(STATUS "   library path: ${${dependency_name}_LIBRARY_DIRS}" )
	MESSAGE(STATUS "   include: ${${dependency_name}_INCLUDE_DIRS}" )
	MESSAGE(STATUS "" )
ENDMACRO (print_dependency dependency_name)

##############################################################################
# find library
##############################################################################
MACRO (find_dependency 
						dependency_name 
						include_search_dirs 
						library_search_dirs 
						header_name 
						lib_name)

IF (NOT ${dependency_name}_INCLUDE_DIRS)

    SET(_${dependency_name}_FOUND_INC_DIRS "")
    FOREACH(_SEARCH_DIR ${include_search_dirs})
        FIND_PATH(_CUR_SEARCH
            NAMES ${header_name}
                PATHS ${_SEARCH_DIR}
                NO_DEFAULT_PATH)
        IF (_CUR_SEARCH)
            LIST(APPEND _${dependency_name}_FOUND_INC_DIRS ${_CUR_SEARCH})
        ENDIF(_CUR_SEARCH)
        SET(_CUR_SEARCH _CUR_SEARCH-NOTFOUND CACHE INTERNAL "internal use")
    ENDFOREACH(_SEARCH_DIR ${${dependency_name}_INCLUDE_SEARCH_DIRS})

    IF (NOT _${dependency_name}_FOUND_INC_DIRS)
        request_search_directories(${dependency_name})
    ENDIF (NOT _${dependency_name}_FOUND_INC_DIRS)
	  
	  FOREACH(_INC_DIR ${_${dependency_name}_FOUND_INC_DIRS})
        SET(${dependency_name}_INCLUDE_DIRS ${${dependency_name}_INCLUDE_DIRS} ${_INC_DIR} CACHE PATH "${dependency_name} include directory.")
    ENDFOREACH(_INC_DIR ${_${dependency_name}_FOUND_INC_DIRS})
    
ENDIF (NOT ${dependency_name}_INCLUDE_DIRS)

IF ( NOT ${dependency_name}_LIBRARY_DIRS )

    SET(_${dependency_name}_FOUND_LIB_DIR "")
    SET(_${dependency_name}_POSTFIX "")

    FOREACH(_SEARCH_DIR ${library_search_dirs})
        FIND_PATH(_CUR_SEARCH
				        NAMES ${lib_name}
                PATHS ${_SEARCH_DIR}
				PATH_SUFFIXES debug release 
                NO_DEFAULT_PATH)
        IF (_CUR_SEARCH)
            LIST(APPEND _${dependency_name}_FOUND_LIB_DIR ${_SEARCH_DIR})
        ENDIF(_CUR_SEARCH)
        SET(_CUR_SEARCH _CUR_SEARCH-NOTFOUND CACHE INTERNAL "internal use")
    ENDFOREACH(_SEARCH_DIR ${${dependency_name}_LIBRARY_SEARCH_DIRS})

    IF (NOT _${dependency_name}_FOUND_LIB_DIR)
        request_search_directories(${dependency_name})
    ELSE (NOT _${dependency_name}_FOUND_LIB_DIR)
		    SET(${dependency_name}_LIBRARY_DIRS ${_${dependency_name}_FOUND_LIB_DIR} CACHE PATH "The ${dependency_name} library directory")
    ENDIF (NOT _${dependency_name}_FOUND_LIB_DIR)
    
    IF (_${dependency_name}_FOUND_LIB_DIR)
        SET(${dependency_name}_LIBRARIES ${lib_name} CACHE FILEPATH "The ${dependency_name} library filename.")
    ENDIF (_${dependency_name}_FOUND_LIB_DIR)
    
ENDIF ( NOT ${dependency_name}_LIBRARY_DIRS )

ENDMACRO (find_dependency dependency_name header_name lib_name)



