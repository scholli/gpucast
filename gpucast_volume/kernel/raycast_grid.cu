/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : grid_raycasting.cu
*  project    : gpucast
*  description:
*
********************************************************************************/

// CUDA includes 
#include <gpucast/volume/cuda_map_resources.hpp> 
 
#include <array>
#include <cmath> 
#include <iostream>
#include <gpucast/gl/util/timer.hpp>

#include <cuda_types.h> 
#include <cuda_globals.h>
#include <device_types.h>

#include <raycast_grid.h>
#include <cuda_renderconfig.h>
#include <get_kernel_workitems.h>

namespace grid
{
  surface<void, cudaSurfaceType2D> out_color_image;
  surface<void, cudaSurfaceType2D> out_depth_image;
  surface<void, cudaSurfaceType2D> in_external_image;
  surface<void, cudaSurfaceType2D> in_position_image;
}

///////////////////////////////////////////////////////////////////////////////
extern "C" __kernel void raycast_grid_kernel ( gpucast::renderconfig  config,
                                               gpucast::bufferinfo    info,
                                               int3                   gridsize,
                                               float4 const*          matrixbuffer,
                                               uint4 const*           gridbuffer,
                                               uint4 const*           facebuffer,
                                               float4 const*          bboxbuffer,
                                               uint4 const*           surfacedatabuffer,
                                               float4 const*          surfacepointsbuffer,
                                               float4 const*          volumedatabuffer,
                                               float4 const*          volumepointsbuffer,
                                               float4 const*          attributedatabuffer,
                                               float2 const*          attributepointsbuffer )
{
  int sx = blockIdx.x*blockDim.x + threadIdx.x;
	int sy = blockIdx.y*blockDim.y + threadIdx.y;

  if ( sx >= config.width || sy >= config.height )
	{ 
		return;
	} 

  int2  coords      = int2_t(sx, sy);

  raycast_grid( config,
                info,
                coords,
                gridsize,
                matrixbuffer,
                gridbuffer,
                facebuffer,
                bboxbuffer,
                surfacedatabuffer,
                surfacepointsbuffer,
                volumedatabuffer,
                volumepointsbuffer,
                attributedatabuffer,
                attributepointsbuffer,
                grid::out_color_image,
                grid::out_depth_image,
                grid::in_position_image,
                grid::in_external_image );
}


///////////////////////////////////////////////////////////////////////////////
extern "C" void invoke_grid_raycasting_kernel ( gpucast::renderconfig           config,
                                                gpucast::bufferinfo             info,
                                                std::array<unsigned, 3> const&  gridsize,
                                                struct cudaGraphicsResource*    input_position_resource,
                                                struct cudaGraphicsResource*    colorbuffer_resource,
                                                struct cudaGraphicsResource*    depthbuffer_resource, 
                                                struct cudaGraphicsResource*    external_color_depth_resource,
                                                struct cudaGraphicsResource*    cuda_grid_buffer,
                                                struct cudaGraphicsResource*    cuda_face_buffer,
                                                struct cudaGraphicsResource*    cuda_bbox_buffer,
                                                struct cudaGraphicsResource*    cuda_surface_data_buffer,
                                                struct cudaGraphicsResource*    cuda_surface_points_buffer,
                                                struct cudaGraphicsResource*    cuda_volume_data_buffer,
                                                struct cudaGraphicsResource*    cuda_volume_points_buffer,
                                                struct cudaGraphicsResource*    cuda_attribute_data_buffer,
                                                struct cudaGraphicsResource*    cuda_attribute_points_buffer,
                                                struct cudaGraphicsResource*    cuda_matrixbuffer )
{ 
  cudaGraphicsResource* cuda_resources[] = {
                                             input_position_resource,
                                             colorbuffer_resource,
                                             depthbuffer_resource,
                                             external_color_depth_resource,
                                             cuda_grid_buffer,
                                             cuda_face_buffer,
                                             cuda_bbox_buffer,
                                             cuda_surface_data_buffer,
                                             cuda_surface_points_buffer,
                                             cuda_volume_data_buffer,
                                             cuda_volume_points_buffer,
                                             cuda_attribute_data_buffer,
                                             cuda_attribute_points_buffer,
                                             cuda_matrixbuffer
                                           };
  
  map_resources ( sizeof ( cuda_resources ) / sizeof ( cudaGraphicsResource* ), cuda_resources );
   
  // map output image
  bind_mapped_resource_to_surface ( colorbuffer_resource,           &grid::out_color_image );
  bind_mapped_resource_to_surface ( depthbuffer_resource,           &grid::out_depth_image );
  bind_mapped_resource_to_surface ( input_position_resource,        &grid::in_position_image );
  bind_mapped_resource_to_surface ( external_color_depth_resource,  &grid::in_external_image );

  uint4*  gridbuffer;
  uint4*  facebuffer;
  float4* bboxbuffer;

  uint4*  surfacedatabuffer;
  float4* surfacepointsbuffer;
  float4* volumedatabuffer;
  float4* volumepointsbuffer;
  float4* attributedatabuffer;
  float2* attributepointsbuffer;
  float4* matrixbuffer;

  int3 gsize = int3_t(gridsize[0], gridsize[1], gridsize[2]);

  bind_mapped_resource_to_pointer ( cuda_grid_buffer,             gridbuffer            );
  bind_mapped_resource_to_pointer ( cuda_face_buffer,             facebuffer            );
  bind_mapped_resource_to_pointer ( cuda_bbox_buffer,             bboxbuffer            );
  
  bind_mapped_resource_to_pointer ( cuda_surface_data_buffer,     surfacedatabuffer     );
  bind_mapped_resource_to_pointer ( cuda_surface_points_buffer,   surfacepointsbuffer   );
  bind_mapped_resource_to_pointer ( cuda_volume_data_buffer,      volumedatabuffer      );
  bind_mapped_resource_to_pointer ( cuda_volume_points_buffer,    volumepointsbuffer    );
  bind_mapped_resource_to_pointer ( cuda_attribute_data_buffer,   attributedatabuffer   );
  bind_mapped_resource_to_pointer ( cuda_attribute_points_buffer, attributepointsbuffer );
  bind_mapped_resource_to_pointer ( cuda_matrixbuffer,            matrixbuffer          );

  { // raycast kernel
    int workitems     = int(get_kernel_workitems ( &raycast_grid_kernel ));

    int workgroups_x  = config.width  + (workitems - config.width % workitems); 
    int workgroups_y  = config.height + (workitems - config.height % workitems);

    dim3 block  ( workitems, workitems, 1 );
    dim3 grid   ( workgroups_x / block.x, workgroups_y / block.y, 1);

    // execute kernel  
    raycast_grid_kernel<<< grid, block>>> ( config,
                                            info,
                                            gsize,
                                            matrixbuffer, 
                                            gridbuffer, 
                                            facebuffer, 
                                            bboxbuffer, 
                                            surfacedatabuffer,
                                            surfacepointsbuffer,
                                            volumedatabuffer,
                                            volumepointsbuffer,
                                            attributedatabuffer,
                                            attributepointsbuffer);
  }

  // unmap gl-resources
  unmap_resources( sizeof ( cuda_resources ) / sizeof ( cudaGraphicsResource* ), cuda_resources );
}
