/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_fragmentlists.cu
*  project    : gpucast
*  description:
*
********************************************************************************/
// CUDA includes 
#include <gpucast/volume/cuda_map_resources.hpp> 
 
#include <cmath> 
#include <iostream>

#include <cuda_types.h> 
#include <cuda_globals.h>
#include <device_types.h>

#include <get_kernel_workitems.h>

namespace passthrough
{
  surface<void, cudaSurfaceType2D> out_color_image;
  surface<void, cudaSurfaceType2D> out_depth_image;
  surface<void, cudaSurfaceType2D> in_external_image;
}

///////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void passthrough_kernel ( int width, int height )
{
  int sx = blockIdx.x*blockDim.x + threadIdx.x;
  int sy = blockIdx.y*blockDim.y + threadIdx.y;

  if ( sx >= width || sy >= height )
  { 
    return;
  }

  int2  coords      = int2_t(sx, sy);

  float4 external_color;
  surf2Dread ( &external_color, passthrough::in_external_image, coords.x*sizeof(float4), coords.y );
  float external_depth = external_color.w;
  external_color.w = 1.0f;

  surf2Dwrite ( external_depth, passthrough::out_depth_image, coords.x*sizeof(float), coords.y );
  surf2Dwrite ( external_color, passthrough::out_color_image, coords.x*sizeof(float4), coords.y );
  
}


///////////////////////////////////////////////////////////////////////////////
extern "C" void invoke_external_passthrough ( unsigned                     width,
                                              unsigned                     height,
                                              struct cudaGraphicsResource* colorbuffer_resource,
                                              struct cudaGraphicsResource* depthbuffer_resource, 
                                              struct cudaGraphicsResource* external_color_depth_resource )
{
  cudaGraphicsResource* cuda_resources[] = { 
                                             colorbuffer_resource,
                                             depthbuffer_resource,
                                             external_color_depth_resource 
                                           };
  
  map_resources ( sizeof ( cuda_resources ) / sizeof ( cudaGraphicsResource* ), cuda_resources );

  bind_mapped_resource_to_surface ( external_color_depth_resource, &passthrough::in_external_image );
  bind_mapped_resource_to_surface ( colorbuffer_resource,          &passthrough::out_color_image );
  bind_mapped_resource_to_surface ( depthbuffer_resource,          &passthrough::out_depth_image );
  
  { // raycast kernel
    std::size_t workitems     = get_kernel_workitems ( &passthrough_kernel );
    
    std::size_t workgroups_x  = width  + (workitems - width % workitems); 
    std::size_t workgroups_y  = height + (workitems - height % workitems);
    
    dim3 block ( workitems, workitems, 1 );
    dim3 grid  ( workgroups_x / block.x, workgroups_y / block.y, 1);
    
    passthrough_kernel<<<grid,block>>>( width, height );
  }

  // unmap gl-resources
  unmap_resources( sizeof ( cuda_resources ) / sizeof ( cudaGraphicsResource* ), cuda_resources );
}

