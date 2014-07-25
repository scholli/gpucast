/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_fragmentlists.cl
*  project    : gpucast
*  description:
*
********************************************************************************/
#include <opencl_types.h>

#include <raycast_fragmentlists.h>

///////////////////////////////////////////////////////////////////////////////
__kernel void run_kernel ( int                    width, 
                           int                    height,
                           float                  nearplane,
                           float                  farplane,
                           float3                 background,
                           int2                   tilesize,
                           uint                   pagesize,
                           uint                   volume_info_offset,
                           float4                 iso_threshold,
                           int                    show_isosides,
                           int                    adaptive_sampling,
                           float                  min_sample_distance,
                           float                  max_sample_distance,
                           float                  adaptive_sample_scale,
                           int                    screenspace_newton_error,
                           float                  fixed_newton_epsilon,
                           uint                   max_iterations_newton,
                           uint                   max_steps_binary_search,
                           float4                 global_attribute_min,
                           float4                 global_attribute_max,
                           float                  surface_transparency,
                           float                  isosurface_transparency,
                           __global float4*       matrices,
                           __global uint4*        indexlist,
                           __global float4*       fraglist,
                           __global float4*       pointbuffer,
                           __global float4*       attributebuffer,
                           __write_only image2d_t colortexture,
                           __write_only image2d_t depthtexture,
                           __read_only  image2d_t headpointer, 
                           __read_only  image2d_t fragmentcount )
{
  /********************************************************************************
  * thread setup
  ********************************************************************************/ 
  int2 coords     = int2_t(get_global_id(0), get_global_id(1));

  if ( coords.x >= width || coords.y >= height )
  {
    return;
  }

  raycast_fragmentlists ( width, 
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
                          fraglist,
                          pointbuffer,
                          attributebuffer,
                          colortexture,
                          depthtexture,
                          headpointer,
                          fragmentcount );
}