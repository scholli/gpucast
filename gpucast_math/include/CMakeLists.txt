###############################################################################
# determine source and header files
###############################################################################
FILE ( GLOB GPUCAST_MATH_MATH_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} gpucast/math/*.hpp )
FILE ( GLOB GPUCAST_MATH_MATH_PARAMETRIC_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} gpucast/math/parametric/*.hpp )
FILE ( GLOB GPUCAST_MATH_MATH_PARAMETRIC_ALGORITHM_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} gpucast/math/parametric/algorithm/*.hpp )
FILE ( GLOB GPUCAST_MATH_MATH_PARAMETRIC_DOMAIN_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} gpucast/math/parametric/domain/*.hpp )
FILE ( GLOB GPUCAST_MATH_MATH_PARAMETRIC_IMPLEMENTATION_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} gpucast/math/parametric/implementation/*.hpp )
FILE ( GLOB GPUCAST_MATH_MATH_UTIL_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} gpucast/math/util.hpp )

SOURCE_GROUP("Header Files" FILES ${GPUCAST_MATH_MATH_INC})
SOURCE_GROUP("Header Files\\parametric" FILES ${GPUCAST_MATH_MATH_PARAMETRIC_INC})
SOURCE_GROUP("Header Files\\parametric\\algorithm" FILES ${GPUCAST_MATH_MATH_PARAMETRIC_ALGORITHM_INC})
SOURCE_GROUP("Header Files\\parametric\\domain" FILES ${GPUCAST_MATH_MATH_PARAMETRIC_DOMAIN_INC})
SOURCE_GROUP("Header Files\\parametric\\implementation" FILES ${GPUCAST_MATH_MATH_PARAMETRIC_IMPLEMENTATION_INC})
SOURCE_GROUP("Header Files\\parametric\\util" FILES ${GPUCAST_MATH_MATH_UTIL_INC})

LINK_DIRECTORIES(${LIB_PATHS})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} include)

ADD_LIBRARY( gpucast_math SHARED
    	     ${GPUCAST_MATH_MATH_INC}
           ${GPUCAST_MATH_MATH_PARAMETRIC_INC}
           ${GPUCAST_MATH_MATH_PARAMETRIC_ALGORITHM_INC}
           ${GPUCAST_MATH_MATH_PARAMETRIC_DOMAIN_INC}
           ${GPUCAST_MATH_MATH_PARAMETRIC_IMPLEMENTATION_INC}
           ${GPUCAST_MATH_MATH_UTIL_INC}
)
SET_TARGET_PROPERTIES(gpucast_math PROPERTIES LINKER_LANGUAGE CXX)

###############################################################################
# install
###############################################################################

# header files 
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION . FILES_MATCHING PATTERN "*.hpp")