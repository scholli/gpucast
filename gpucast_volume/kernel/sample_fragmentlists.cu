/********************************************************************************
*
* Copyright (C) 2009-2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : sample_fragmentlists.cu
*  project    : gpucast
*  description:
*
********************************************************************************/
#include <cmath> 
#include <iostream>

#include <gpucast/volume/cuda_map_resources.hpp> 

// C cuda includes
#include <cuda_types.h> 
#include <device_types.h>

#include <get_kernel_workitems.h>
#include <sample_fragmentlists.h>

// project includes

#include <gpucast/gl/util/timer.hpp>
  /*
///////////////////////////////////////////////////////////////////////////////
extern "C" __kernel void sort_kernel    ( int         width, 
                                          int         height,
                                          int2        tilesize,
                                          unsigned    pagesize,
                                          uint4*      indexlist )
{
  int sx = blockIdx.x*blockDim.x + threadIdx.x;
	int sy = blockIdx.y*blockDim.y + threadIdx.y;

  if ( sx >= width || sy >= height )
	{ 
		return;
	} 

  int2  coords        = int2_t(sx, sy);

  int fragindex       = 0; 
  unsigned nfragments = 0;

  surf2Dread ( &fragindex,  fraglist::in_headpointer_image, coords.x*sizeof(float), coords.y );
  surf2Dread ( &nfragments, fraglist::in_fragmentcount_image, coords.x*sizeof(float), coords.y );

  //uint nfragments = count_fragments(indexlist, fragindex);
   
  bubble_sort_indexlist_1 ( fragindex, nfragments, indexlist ); 
}
 */

namespace unified_sampling
{
  surface<void, cudaSurfaceType2D> out_color_image;
  surface<void, cudaSurfaceType2D> out_depth_image;
  surface<void, cudaSurfaceType2D> in_headpointer_image;
  surface<void, cudaSurfaceType2D> in_fragmentcount_image;
  surface<void, cudaSurfaceType2D> in_external_image;
}

///////////////////////////////////////////////////////////////////////////////
extern "C" __kernel void sample_fragmentlist_kernel (  int            width,  
                                                       int            height,
                                                       int2           tilesize,
                                                       unsigned       pagesize,
                                                       unsigned       volume_info_offset,
                                                       float          nearplane,
                                                       float          farplane,
                                                       float3         background, 
                                                       float          iso_threshold,
                                                       int            show_isosides,
                                                       int            adaptive_sampling,
                                                       float          min_sample_distance,
                                                       float          max_sample_distance,
                                                       float          adaptive_sample_scale,
                                                       int            screenspace_newton_error,
                                                       float          fixed_newton_epsilon, 
                                                       unsigned       max_iterations_newton,
                                                       unsigned       max_steps_binary_search,
                                                       float          global_attribute_min,
                                                       float          global_attribute_max,  
                                                       float          surface_transparency, 
                                                       float          isosurface_transparency, 
                                                       float4 const*  matrices, 
                                                       uint4*         indexlist,
                                                       uint4 const*   surfacedatabuffer, 
                                                       float4 const*  surfacepointsbuffer, 
                                                       float4 const*  volumedatabuffer, 
                                                       float4 const*  volumepointsbuffer, 
                                                       float4 const*  attributedatabuffer,
                                                       float2 const*  attributepointsbuffer
                                          )
{
  int sx = blockIdx.x*blockDim.x + threadIdx.x;
	int sy = blockIdx.y*blockDim.y + threadIdx.y;

  if ( sx >= width || sy >= height )
	{ 
		return;
	}

  int2  coords      = int2_t(sx, sy);

  clock_t t0 = clock();  
    
  sample_fragmentlists ( threadIdx.x,
                         width, 
                         height, 
                         coords,  
                         nearplane,
                         farplane,
                         background,
                         tilesize,
                         pagesize,
                         volume_info_offset,
                         iso_threshold,
                         show_isosides,
                         adaptive_sampling,
                         min_sample_distance,
                         max_sample_distance, 
                         adaptive_sample_scale,
                         screenspace_newton_error,
                         fixed_newton_epsilon, 
                         max_iterations_newton,
                         max_steps_binary_search,
                         global_attribute_min, 
                         global_attribute_max, 
                         surface_transparency, 
                         isosurface_transparency,
                         matrices, 
                         indexlist,  
                         surfacedatabuffer,
                         surfacepointsbuffer,
                         volumedatabuffer, 
                         volumepointsbuffer, 
                         attributedatabuffer, 
                         attributepointsbuffer, 
                         unified_sampling::out_color_image,
                         unified_sampling::out_depth_image,
                         unified_sampling::in_headpointer_image,  
                         unified_sampling::in_external_image );
     
  clock_t t1 = clock();  
   
  float relative_costs = float_t(t1 - t0) / 10000000.0;   
       
  float4 costs_color = transferfunction ( relative_costs );  
  //surf2Dwrite ( costs_color, out_color_image, coords.x*sizeof(float4), coords.y );
}   


///////////////////////////////////////////////////////////////////////////////
extern "C" void invoke_unified_sampling  ( unsigned                     width,
                                           unsigned                     height,
                                           int2                         tilesize, 
                                           unsigned                     pagesize,
                                           unsigned                     volume_info_offset,
                                           float                        nearplane,
                                           float                        farplane,
                                           float3                       background, 
                                           float                        iso_threshold,
                                           int                          show_isosides,
                                           int                          adaptive_sampling,
                                           float                        min_sample_distance, 
                                           float                        max_sample_distance,
                                           float                        adaptive_sample_scale,
                                           int                          screenspace_newton_error,  
                                           float                        fixed_newton_epsilon,  
                                           unsigned                     max_iterations_newton,
                                           unsigned                     max_steps_binary_search,
                                           float                        global_attribute_min,
                                           float                        global_attribute_max,
                                           float                        surface_transparency,
                                           float                        isosurface_transparency,  
                                           struct cudaGraphicsResource* matrices_resource,
                                           struct cudaGraphicsResource* colorbuffer_resource,
                                           struct cudaGraphicsResource* depthbuffer_resource, 
                                           struct cudaGraphicsResource* headpointer_resource, 
                                           struct cudaGraphicsResource* fragmentcount_resource,  
                                           struct cudaGraphicsResource* indexlist_resource, 
                                           struct cudaGraphicsResource* surface_data_buffer_resource,
                                           struct cudaGraphicsResource* surface_points_buffer_resource,
                                           struct cudaGraphicsResource* volume_data_buffer_resource,
                                           struct cudaGraphicsResource* volume_points_buffer_resource,
                                           struct cudaGraphicsResource* attribute_data_buffer_resource,
                                           struct cudaGraphicsResource* attribute_points_buffer_resource,
                                           struct cudaGraphicsResource* external_color_depth_resource )
{ 
  cudaGraphicsResource* cuda_resources[] = { matrices_resource     ,
                                             colorbuffer_resource  ,
                                             depthbuffer_resource  ,
                                             headpointer_resource  ,
                                             fragmentcount_resource,
                                             indexlist_resource    ,
                                             surface_data_buffer_resource,
                                             surface_points_buffer_resource,
                                             volume_data_buffer_resource, 
                                             volume_points_buffer_resource,
                                             attribute_data_buffer_resource,
                                             attribute_points_buffer_resource,
                                             external_color_depth_resource
                                           };

  map_resources ( sizeof ( cuda_resources ) / sizeof ( cudaGraphicsResource* ), cuda_resources );
   
  // map output image
  bind_mapped_resource_to_surface ( colorbuffer_resource,           &unified_sampling::out_color_image );
  bind_mapped_resource_to_surface ( depthbuffer_resource,           &unified_sampling::out_depth_image );
  bind_mapped_resource_to_surface ( headpointer_resource,           &unified_sampling::in_headpointer_image );
  bind_mapped_resource_to_surface ( fragmentcount_resource,         &unified_sampling::in_fragmentcount_image );
  bind_mapped_resource_to_surface ( external_color_depth_resource,  &unified_sampling::in_external_image );

  // retrieve device pointer for mapped buffers  
  uint4*      indexlist;
  float4*     matrices; 
  uint4*      surfacedatabuffer; 
  float4*     surfacepointsbuffer;
  float4*     volumedatabuffer; 
  float4*     volumepointsbuffer; 
  float4*     attributedatabuffer; 
  float2*     attributepointsbuffer;

  bind_mapped_resource_to_pointer ( matrices_resource        , matrices        );
  bind_mapped_resource_to_pointer ( indexlist_resource       , indexlist       );

  bind_mapped_resource_to_pointer ( surface_data_buffer_resource     , surfacedatabuffer     );
  bind_mapped_resource_to_pointer ( surface_points_buffer_resource   , surfacepointsbuffer   );
  bind_mapped_resource_to_pointer ( volume_data_buffer_resource      , volumedatabuffer      );
  bind_mapped_resource_to_pointer ( volume_points_buffer_resource    , volumepointsbuffer    );
  bind_mapped_resource_to_pointer ( attribute_data_buffer_resource   , attributedatabuffer   );
  bind_mapped_resource_to_pointer ( attribute_points_buffer_resource , attributepointsbuffer );

  float memsettime;

  cudaEvent_t start, stop; 
   
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // single sort pass
#if 0 
  cudaEventRecord(start,0); 
  cudaThreadSynchronize();

  { // sort kernel
    std::size_t workitems     = get_kernel_workitems ( "sort_kernel" );

    std::size_t workgroups_x  = width  + (workitems - width % workitems); 
    std::size_t workgroups_y  = height + (workitems - height % workitems);

    dim3 block  ( workitems, workitems, 1 );
    dim3 grid   ( workgroups_x / block.x, workgroups_y / block.y, 1);

    sort_kernel<<< grid, block>>> ( width,  
                                    height, 
                                    tilesize,
                                    pagesize, 
                                    indexlist );
  } 

  cudaEventRecord(stop,0); 
  cudaThreadSynchronize();

  cudaEventElapsedTime(&memsettime, start, stop); 
  std::cout << " sort kernel ms : " << memsettime; 
#endif

  cudaEventRecord(start,0); 

  { // raycast kernel
    std::size_t workitems     = get_kernel_workitems ( &sample_fragmentlist_kernel );
    //std::size_t workitems     = 8;

    //std::cout << workitems << " workitems per block.\n";

    std::size_t workgroups_x  = width  + (workitems - width % workitems); 
    std::size_t workgroups_y  = height + (workitems - height % workitems);

    dim3 block  ( workitems, workitems, 1 );
    dim3 grid   ( workgroups_x / block.x, workgroups_y / block.y, 1);

     
    // execute kernel  
    sample_fragmentlist_kernel<<< grid, block>>> ( width,  
                                                   height, 
                                                   tilesize,
                                                   pagesize,
                                                   volume_info_offset, 
                                                   nearplane,  
                                                   farplane,
                                                   background, 
                                                   iso_threshold,
                                                   show_isosides,
                                                   adaptive_sampling, 
                                                   min_sample_distance, 
                                                   max_sample_distance,  
                                                   adaptive_sample_scale,
                                                   screenspace_newton_error,
                                                   fixed_newton_epsilon,
                                                   max_iterations_newton,
                                                   max_steps_binary_search,
                                                   global_attribute_min,
                                                   global_attribute_max,
                                                   surface_transparency,
                                                   isosurface_transparency,   
                                                   matrices, 
                                                   indexlist,
                                                   surfacedatabuffer, 
                                                   surfacepointsbuffer, 
                                                   volumedatabuffer, 
                                                   volumepointsbuffer,
                                                   attributedatabuffer,
                                                   attributepointsbuffer);

  } 

  cudaEventRecord(stop, 0);   
  cudaThreadSynchronize();

  cudaEventElapsedTime(&memsettime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // unmap gl-resources
  unmap_resources( sizeof ( cuda_resources ) / sizeof ( cudaGraphicsResource* ), cuda_resources );
}



 