include(FindUnitTest++)

file(GLOB GPUCAST_MATH_UNITTEST_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

LINK_DIRECTORIES(${LIB_PATHS} ${UNITTEST_LIBRARY_DIRS})

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${UNITTEST++_INCLUDE_DIR} 
                     ../include
                   )

ADD_EXECUTABLE( gpucast_math_unittests ${GPUCAST_MATH_UNITTEST_SRC})

IF (MSVC)
  TARGET_LINK_LIBRARIES( gpucast_math_unittests 
                         debug ${UNITTEST++_LIBRARY_DEBUG} optimized ${UNITTEST++_LIBRARY})
ELSEIF (UNIX)
  TARGET_LINK_LIBRARIES( gpucast_math_unittests ${UNITTEST++_LIBRARY_DEBUG})
ENDIF (MSVC)
