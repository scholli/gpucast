/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_fragmentlists_for_iso_surface.h
*  project    : gpucast
*  description:
*
********************************************************************************/

#ifdef CUDA_BACKEND
  #include "cuda_types.h"
#else
  #include "opencl_types.h"
#endif

#include "./local_memory_config.h"

#include "fragmentlist/count_fragments.h"
#include "fragmentlist/search_exit_fragment.h"

#include "math/adjugate.h"
#include "math/compute_depth.h"
#include "math/conversion.h"
#include "math/determinant.h"
#include "math/horner_surface.h"
#include "math/horner_volume.h"
#include "math/inverse.h"
#include "math/mult.h"
#include "math/newton_surface.h"
#include "math/newton_volume.h"
#include "math/parameter_on_domain_boundary.h"

#include "isosurface/compute_sampling_position.h"
#include "isosurface/target_function.h"
#include "isosurface/validate_bezier_domain.h"
#include "isosurface/validate_isosurface_intersection.h"
#include "isosurface/search_volume_for_iso_surface.h"

#include "shade/phong.h"


///////////////////////////////////////////////////////////////////////////////
__kernel void run_kernel ( int                    width, 
                           int                    height,
                           float                  nearplane,
                           float                  farplane,
                           int2 const&            tilesize,
                           uint                   pagesize,
                           uint                   volume_info_offset,
                           float4 const&          iso_threshold,
                           int                    show_isosides,
                           int                    adaptive_sampling,
                           float                  min_sample_distance,
                           float                  max_sample_distance,
                           float                  adaptive_sample_scale,
                           int                    screenspace_newton_error,
                           float                  fixed_newton_epsilon,
                           uint                   max_iterations_newton,
                           uint                   max_steps_binary_search,
                           float4 const&          global_attribute_min,
                           float4 const&          global_attribute_max,
                           float                  surface_transparency,
                           __global float4*       matrices,
                           __global uint4*        indexlist,
                           __global float4*       fraglist,
                           __global float4*       pointbuffer,
                           __global float4*       attributebuffer,
                           __write_only image2d_t colortexture,
                           __write_only image2d_t depthtexture)
{
  /********************************************************************************
  * thread setup
  ********************************************************************************/ 
  int gid_x       = get_group_id (0);
  int gid_y       = get_group_id (1);
  
  int lid_x       = get_local_id (0);
  int lid_y       = get_local_id (1);
  
  int size_x      = get_global_size (0);
  int size_y      = get_global_size (1);
  
  int groupsize_x = get_local_size (0);
  int groupsize_y = get_local_size (1);

  int2 coords     = (int2)( get_group_id (0) * get_local_size (0) + get_local_id (0), 
                            get_group_id (1) * get_local_size (1) + get_local_id (1) );

  int pixel_index = 0;

  int2 resolution      = (int2)(width, height);
  int2 tile_resolution = (int2)(resolution / tilesize) + (int2)(clamp(resolution%tilesize, (int2)(0,0), (int2)(1,1)));

  int2 tile_id         = (int2)(coords%tilesize);
  int2 tile_coords     = coords / tilesize;

  int chunksize         = tilesize.x * tilesize.y * pagesize;

  pixel_index           = tile_coords.y * tile_resolution.x * chunksize + 
                          tile_coords.x * chunksize + 
                          tile_id.y * tilesize.x * pagesize + 
                          tile_id.x * pagesize;

  /********************************************************************************
  * allocate variables for ray casting result
  ********************************************************************************/ 
  int    fragindex                   = pixel_index;

  float3 iso_surface_uvw             = (float3)(0,0,0);
  float4 iso_surface_position        = (float4)(0,0,0,0);
  float4 iso_surface_normal          = (float4)(0,0,0,0);
  float4 iso_surface_attribute       = (float4)(0,0,0,0);

  float  current_depth               = MAXFLOAT;
  float4 last_sample_position        = (float4)(0,0,0,0);
  float4 attrib                      = (float4)(0,0,0,0);
  float4 adu                         = (float4)(0,0,0,0); 
  float4 adv                         = (float4)(0,0,0,0); 
  float4 adw                         = (float4)(0,0,0,0);

  float4 p                           = (float4)(0,0,0,0);
  float4 du                          = (float4)(0,0,0,0);
  float4 dv                          = (float4)(0,0,0,0);
  float4 dw                          = (float4)(0,0,0,0);

  /********************************************************************************
  * init out color and transparency
  ********************************************************************************/ 
  float4 out_color                   = (float4)(0,0,0,0);
  float  out_opacity                 = 1.0f;
  float const isosurface_opacity     = surface_transparency;

  /********************************************************************************
  * write only pixels within framebuffer
  ********************************************************************************/ 
  if ( coords.x < width && coords.y < height ) 
  {
    uint nfragments = count_fragments(indexlist, pixel_index);
    if ( nfragments > 0 ) 
    {
      /********************************************************************************
      * create transform matrices
      ********************************************************************************/
      float4 modelviewmatrix[4]                  = {*(matrices    ),  *(matrices + 1),  *(matrices + 2),  *(matrices + 3)};
      float4 modelviewmatrixinverse[4]           = {*(matrices + 4),  *(matrices + 5),  *(matrices + 6),  *(matrices + 7)};
      float4 modelviewprojectionmatrix[4]        = {*(matrices + 8),  *(matrices + 9),  *(matrices + 10), *(matrices + 11)};
      float4 normalmatrix[4]                     = {*(matrices + 12), *(matrices + 13), *(matrices + 14), *(matrices + 15)};
      float4 modelviewprojectionmatrixinverse[4] = {*(matrices + 16), *(matrices + 17), *(matrices + 18), *(matrices + 19)};

      /********************************************************************************
      * traverse depth-sorted fragments
      ********************************************************************************/ 
      for ( int i = 0; i != nfragments; ++i )
      {
        /********************************************************************************
        * early ray termination
        ********************************************************************************/ 
        if ( out_opacity < 0.01 )
        {
          break;
        }

        /********************************************************************************
        * retrieve information about fragment, volume and surface
        ********************************************************************************/ 
        uint4   fragindexinfo = indexlist [ fragindex ];
        float4  fragmentdata0 = fraglist  [ fragindexinfo.z    ];
        float4  fragmentdata1 = fraglist  [ fragindexinfo.z + 1];
        float4  fragposition  = (float4)(fragmentdata1.xyz, 1.0);

        float3  start_uvw     = fragmentdata0.xyz;
        int     volume_id     = fragindexinfo.y;
        int     surface_id    = floatBitsToInt(fragmentdata1.w);
        float4  volume_info   = pointbuffer[ volume_id ];
        float4  volume_info2  = pointbuffer[ volume_id + 1 ];

        int     attribute_id  = (int)(volume_info.z);
        uint3   order         = (uint3)((uint)(volume_info2.x), (uint)(volume_info2.y), (uint)(volume_info2.z));
        float   bbox_size     = volume_info.w;

        // could prefetch, but results not faster
        //prefetch ( pointbuffer + volume_id        + volume_info_offset, order.x * order.y * order.z );
        //prefetch ( attributebuffer + attribute_id + volume_info_offset, order.x * order.y * order.z );

        /********************************************************************************
        * determine if ray enters volume at this fragment and search for exit fragment
        ********************************************************************************/   
        uint4 fragment_volume_exit   = (uint4)(0);
        uint  nfragments_found       = 0;
        bool  search_for_iso_surface = search_volume_exit_fragment ( indexlist, 
                                                                     volume_id, 
                                                                     fragindexinfo.x, 
                                                                     (int)(nfragments - i - 1), 
                                                                     &fragment_volume_exit, 
                                                                     &nfragments_found );
        

        /********************************************************************************
        * ray setup in object coordinates
        ********************************************************************************/
        float4 ray_entry        = fragposition;
        float4 ray_exit         = fraglist [fragment_volume_exit.z + 1];

        /********************************************************************************
        * if ray enters volume -> determine if there is an intersection with an iso surface
        ********************************************************************************/   
        if ( search_for_iso_surface )
        {
          bool continue_isosurface_search                   = true;
          int isosurface_intersections_per_volume           = 0;
          int const max_isosurface_intersections_per_volume = 3;

          while ( continue_isosurface_search && isosurface_intersections_per_volume < max_isosurface_intersections_per_volume )
          {
            ++isosurface_intersections_per_volume;

            float4 debug;
            continue_isosurface_search = search_volume_for_iso_surface (pointbuffer,
                                                                        attributebuffer,
                                                                        volume_id + volume_info_offset,
                                                                        attribute_id + volume_info_offset,
                                                                        order,
                                                                        start_uvw,
                                                                        iso_threshold,
                                                                        ray_entry,
                                                                        ray_exit,
                                                                        adaptive_sampling,
                                                                        bbox_size,
                                                                        min_sample_distance,
                                                                        max_sample_distance,
                                                                        adaptive_sample_scale,
                                                                        screenspace_newton_error,
                                                                        fixed_newton_epsilon,
                                                                        max_iterations_newton,
                                                                        max_steps_binary_search,
                                                                        &iso_surface_position,
                                                                        &iso_surface_attribute,
                                                                        &iso_surface_normal,
                                                                        &iso_surface_uvw,
                                                                        &ray_entry,
                                                                        &start_uvw,
                                                                        &debug
                                                                        );
         
            if ( continue_isosurface_search )
            {
              // shade
              float4 lightpos    = (float4)(0.0f, 0.0f, 0.0f, 1.0f); // light from camera
              float4 pworld      = mult_mat4_float4 ( modelviewmatrix, iso_surface_position );
              iso_surface_normal = mult_mat4_float4 ( normalmatrix, iso_surface_normal );

                            
              current_depth      = compute_depth_from_world_coordinates ( pworld, nearplane, farplane );
              float3 L           = normalize ( lightpos.xyz - pworld.xyz );
              float3 N           = normalize ( iso_surface_normal.xyz );
              N                  = faceforward ( -pworld.xyz, N );
              float diffuse      = dot (N , L);
              diffuse            = ( diffuse * 0.5f ) + 0.5f;

              //out_color          = isosurface_opacity * diffuse * (float4)((iso_surface_attribute.xyz - global_attribute_min.xyz) / (global_attribute_max.xyz - global_attribute_min.xyz), 1.0);;
              out_color         += isosurface_opacity * out_opacity * diffuse * (float4)((iso_surface_attribute.xyz - global_attribute_min.xyz) / (global_attribute_max.xyz - global_attribute_min.xyz), 1.0);;
              out_opacity       *= (1.0f - isosurface_opacity);
              //out_color         = (float4)(N,1.0);
              //break;
            } 
          } // while there still might be an isosurface transition in interval
        } // found_exit fragment -> search interval for iso surface

        // go to next fragment
        fragindex = fragindexinfo.x;
      } // for all fragments
      write_imagef ( depthtexture, coords, current_depth );
      write_imagef ( colortexture, coords, out_color );
    } else {
      write_imagef ( colortexture, coords, (float4)(0,0,0,0) );
    } // if there are fragments in indexlist
  } // if in screen: [x,y] in [width, height]
}