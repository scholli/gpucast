###############################################################################
# determine source and header files
###############################################################################
FILE ( GLOB GPUCAST_VOLUME_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/*.hpp )
FILE ( GLOB GPUCAST_VOLUME_IMPORT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/import/*.hpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/isosurface/*.hpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_FRAGMENT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/isosurface/fragment/*.hpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_GRID_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/isosurface/grid/*.hpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_SPLAT_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/isosurface/splat/*.hpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_OCTREE_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/gpucast/volume/isosurface/octree/*.hpp )

FILE ( GLOB GPUCAST_VOLUME_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp )
FILE ( GLOB GPUCAST_VOLUME_IMPORT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/import/*.cpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/isosurface/*.cpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_FRAGMENT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/isosurface/fragment/*.cpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_GRID_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/isosurface/grid/*.cpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_SPLAT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/isosurface/splat/*.cpp )
FILE ( GLOB GPUCAST_VOLUME_ISOSURFACE_OCTREE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/isosurface/octree/*.cpp )

FILE ( GLOB_RECURSE GPUCAST_VOLUME_CUDA_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} kernel/*.cu )
FILE ( GLOB_RECURSE GPUCAST_VOLUME_CUDA_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} kernel/*.h )

SOURCE_GROUP("Header Files" FILES                       ${GPUCAST_VOLUME_INC})
SOURCE_GROUP("Header Files\\import" FILES               ${GPUCAST_VOLUME_IMPORT_INC})
SOURCE_GROUP("Header Files\\isosurface" FILES           ${GPUCAST_VOLUME_ISOSURFACE_INC})
SOURCE_GROUP("Header Files\\isosurface\\fragment" FILES ${GPUCAST_VOLUME_ISOSURFACE_FRAGMENT_INC})
SOURCE_GROUP("Header Files\\isosurface\\grid"     FILES ${GPUCAST_VOLUME_ISOSURFACE_GRID_INC})
SOURCE_GROUP("Header Files\\isosurface\\splat"    FILES ${GPUCAST_VOLUME_ISOSURFACE_SPLAT_INC})
SOURCE_GROUP("Header Files\\isosurface\\octree"   FILES ${GPUCAST_VOLUME_ISOSURFACE_OCTREE_INC})

SOURCE_GROUP("Source Files" FILES                       ${GPUCAST_VOLUME_SRC})
SOURCE_GROUP("Source Files\\import" FILES               ${GPUCAST_VOLUME_IMPORT_SRC})
SOURCE_GROUP("Source Files\\isosurface" FILES           ${GPUCAST_VOLUME_ISOSURFACE_SRC})
SOURCE_GROUP("Source Files\\isosurface\\fragment" FILES ${GPUCAST_VOLUME_ISOSURFACE_FRAGMENT_SRC})
SOURCE_GROUP("Source Files\\isosurface\\grid"     FILES ${GPUCAST_VOLUME_ISOSURFACE_GRID_SRC})
SOURCE_GROUP("Source Files\\isosurface\\splat"    FILES ${GPUCAST_VOLUME_ISOSURFACE_SPLAT_SRC})
SOURCE_GROUP("Source Files\\isosurface\\octree"   FILES ${GPUCAST_VOLUME_ISOSURFACE_OCTREE_SRC})

SOURCE_GROUP("Source Files\\CUDA"   FILES ${GPUCAST_VOLUME_CUDA_SRC})
SOURCE_GROUP("Header Files\\CUDA"   FILES ${GPUCAST_VOLUME_CUDA_INC})

LINK_DIRECTORIES(${LIB_PATHS} ${BOOST_LIBRARY_DIRS} ${QHULL_LIBRARY_DIRS} ${FREEIMAGE_LIBRARY_DIRS})

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -Xcompiler ,\"/MD\")
SET(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS} )
SET(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS} )

CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include
                         ${CMAKE_CURRENT_SOURCE_DIR}/kernel
                         ${CMAKE_SOURCE_DIR}/gpucast_gl/include
                         ${CMAKE_SOURCE_DIR}/gpucast_math/include
                         ${CMAKE_SOURCE_DIR}/gpucast_core/include
                         ${CMAKE_SOURCE_DIR}/externals/include/glew)

SET(GPUCAST_VOLUME_PASSTHROUGH_SRC kernel/external_passthrough.cu)
SET(CUDA_VERBOSE_BUILD ON)

SET(CUDA_PASSTHROUGH_KERNEL_SRC kernel/external_passthrough.cu)
SET(CUDA_RAYCAST_FRAGLISTS_KERNEL_SRC kernel/raycast_fragmentlists.cu)
SET(CUDA_RAYCAST_GRID_KERNEL_SRC kernel/raycast_grid.cu)
SET(CUDA_RAYCAST_OCTREE_KERNEL_SRC kernel/raycast_octree.cu)
SET(CUDA_VOLUME_SPLAT_KERNEL_SRC kernel/splat_volume_samples.cu)

CUDA_ADD_LIBRARY( gpucast_volume SHARED
             ${GPUCAST_VOLUME_SRC}
             ${GPUCAST_VOLUME_IMPORT_SRC}
             ${GPUCAST_VOLUME_ISOSURFACE_SRC}
             ${GPUCAST_VOLUME_ISOSURFACE_FRAGMENT_SRC}
             ${GPUCAST_VOLUME_ISOSURFACE_GRID_SRC}
             ${GPUCAST_VOLUME_ISOSURFACE_SPLAT_SRC}
             ${GPUCAST_VOLUME_ISOSURFACE_OCTREE_SRC}
             ${GPUCAST_VOLUME_INC}
             ${GPUCAST_VOLUME_IMPORT_INC}
             ${GPUCAST_VOLUME_ISOSURFACE_INC}
             ${GPUCAST_VOLUME_ISOSURFACE_FRAGMENT_INC}
             ${GPUCAST_VOLUME_ISOSURFACE_GRID_INC}
             ${GPUCAST_VOLUME_ISOSURFACE_SPLAT_INC}
             ${GPUCAST_VOLUME_ISOSURFACE_OCTREE_INC}
             ${CUDA_PASSTHROUGH_KERNEL_SRC}
             ${CUDA_RAYCAST_FRAGLISTS_KERNEL_SRC}
             ${CUDA_RAYCAST_GRID_KERNEL_SRC}
             ${CUDA_RAYCAST_OCTREE_KERNEL_SRC}
             ${CUDA_VOLUME_SPLAT_KERNEL_SRC}
)

INCLUDE_DIRECTORIES( ${INCLUDE_PATHS} 
                     ${GLEW_INCLUDE_DIRS}
                     ${QHULL_INCLUDE_DIRS} 
                     ${CUDA_INCLUDE_DIRS}
                     ../gpucast_core/include
                     ../gpucast_gl/include
                     ../gpucast_math/include
                     include
                     kernel )

###############################################################################
# set preprocessor configuration
###############################################################################
SET (GPUCAST_VOLUME_BUILD_FLAGS ${BUILD_FLAGS})

IF (MSVC)
  SET(GPUCAST_VOLUME_BUILD_FLAGS "${GPUCAST_VOLUME_BUILD_FLAGS} -D GPUCAST_VOLUME_EXPORT")
ENDIF (MSVC)

SET_TARGET_PROPERTIES( gpucast_volume PROPERTIES COMPILE_FLAGS ${GPUCAST_VOLUME_BUILD_FLAGS})

###############################################################################
# dependencies
###############################################################################
ADD_DEPENDENCIES(gpucast_volume gpucast_math gpucast_core gpucast_gl)

TARGET_LINK_LIBRARIES( gpucast_volume 
                        ${LIBRARIES} 
                        ${CUDA_LIBRARIES}
                        gpucast_core 
                        gpucast_gl
)
