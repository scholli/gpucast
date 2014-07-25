/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_fragmentlists.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_RAYCAST_FRAGMENTLISTS_H
#define GPUCAST_RAYCAST_FRAGMENTLISTS_H

#include <math/adjugate.h>
#include <math/compute_depth.h>
#include <math/constants.h>
#include <math/conversion.h>
#include <math/determinant.h>
#include <math/horner_surface.h>
#include <math/horner_volume.h>
#include <math/inverse.h>
#include <math/in_domain.h>
#include <math/mult.h>
#include <math/newton_surface.h>
#include <math/newton_volume.h>
#include <math/normalize.h>
#include <math/parameter_on_domain_boundary.h>
#include <math/transferfunction.h>

#include <fragmentlist/bubble_sort.h>
#include <fragmentlist/classify_intersection.h>
#include <fragmentlist/classify_transition.h>
#include <fragmentlist/count_fragments.h>
#include <fragmentlist/compute_indexlistindex.h>
#include <fragmentlist/find_ray_exit.h>
#include <fragmentlist/load_volume_data.h>
#include <fragmentlist/search_exit_fragment.h>
#include <fragmentlist/raycast_surfaces_in_fragmentlist.h>

#include <isosurface/compute_sampling_position.h>
#include <isosurface/target_function.h>
#include <isosurface/fragment.h>
#include <isosurface/history.h>
#include <isosurface/face_info.h>
#include <isosurface/ray_state.h>
#include <isosurface/volume_info.h>
#include <isosurface/validate_bezier_domain.h>
#include <isosurface/validate_isosurface_intersection.h>
#include <isosurface/search_volume_for_iso_surface.h>

#include <shade/phong.h>

#define SEARCH_FOR_ISOSURFACE           1
#define SHADE_OUTER_SURFACE             1
#define PER_FRAGMENT_INTERSECTION       0

__device__ inline unsigned 
determine_contained_volume ( face_info const& last_face, face_info const& current_face )
{
  if ( last_face.volume_data_id == current_face.volume_data_id ) {
    return last_face.volume_data_id;
  }

  if ( last_face.adjacent_volume_data_id == current_face.volume_data_id )
  {
    return last_face.adjacent_volume_data_id;
  }

  if ( last_face.volume_data_id == current_face.adjacent_volume_data_id )
  {
    return current_face.adjacent_volume_data_id;
  }

  if ( last_face.adjacent_volume_data_id == current_face.adjacent_volume_data_id )
  {
    return current_face.adjacent_volume_data_id;
  } else {
    return 0;
  }
}

__device__
void raycast_fragmentlists ( int                      threadidx,
                             int                      width, 
                             int                      height,
                             int2 const&              coords,
                             float                    nearplane,
                             float                    farplane,
                             float3 const&            background,
                             int2 const&              tilesize,
                             unsigned                 pagesize,
                             unsigned                 backface_culling,
                             float                    iso_threshold,
                             int                      adaptive_sampling,
                             float                    min_sample_distance,
                             float                    max_sample_distance,
                             float                    adaptive_sample_scale,
                             int                      screenspace_newton_error,
                             float                    fixed_newton_epsilon,
                             unsigned                 max_iterations_newton,
                             unsigned                 max_steps_binary_search,
                             float                    global_attribute_min,
                             float                    global_attribute_max,
                             float                    surface_transparency,
                             float                    isosurface_transparency,
                             bool                     show_samples_isosurface_intersection,
                             bool                     show_samples_for_face_intersection,
                             bool                     show_face_intersections,
                             bool                     show_face_intersection_tests,
                             bool                     show_isosides,
                             bool                     detect_face_by_sampling,
                             bool                     detect_implicit_extremum,
                             bool                     detect_implicit_inflection,
                             float4 const*            matrices,
                             uint4*                   indexlist,
                             unsigned*                allocation_grid,
                             uint4 const*             surfacedatabuffer,
                             float4 const*            surfacepointsbuffer,
                             float4 const*            volumedatabuffer,
                             float4 const*            volumepointsbuffer,
                             float4 const*            attributedatabuffer,
                             float2 const*            attributepointsbuffer,
                             __write_only image2d_t   colortexture,
                             __write_only image2d_t   depthtexture,
                             __read_only image2d_t    headpointer_image,
                             __read_only image2d_t    external_color_depth_image )
{
  /********************************************************************************
  * allocate variables for ray casting result
  ********************************************************************************/ 
  float  current_depth = MAX_FLOAT;
  int    fragindex     = 0;
  surf2Dread ( &fragindex, headpointer_image, coords.x*sizeof(float), coords.y );

  /********************************************************************************
  * init out color and transparency
  ********************************************************************************/ 
  int    total_samples               = 0;

  /********************************************************************************
  * write only pixels within framebuffer
  ********************************************************************************/ 
  unsigned nfragments               = count_fragments(indexlist, fragindex);
  unsigned npoint_evaluations       = 0;
  unsigned nface_intersection_tests = 0;
  unsigned nface_intersections      = 0;

  float4 external_color;
  surf2Dread ( &external_color, external_color_depth_image, coords.x*sizeof(float4), coords.y );
  float external_depth = external_color.w;
  external_color.w = 1.0f;
  
  /********************************************************************************
  * first: if there are no fragments - use external color
  ********************************************************************************/
  if ( nfragments == 0 )
  {
    if ( external_depth < 1.0f ) 
    {
      surf2Dwrite ( external_depth, depthtexture, coords.x*sizeof(float), coords.y );
      surf2Dwrite ( external_color, colortexture, coords.x*sizeof(float4), coords.y );
    }
    return;
  }

  /********************************************************************************
  * init memory for matrices
  ********************************************************************************/  
  float4 const* modelviewmatrix                 = matrices;
  float4 const* modelviewmatrixinverse          = matrices + 4;
  //float4 modelviewmatrixinverse[4]          ;// = {*(matrices + 4),  *(matrices + 5),  *(matrices + 6),  *(matrices + 7)};
  //float4 modelviewprojectionmatrix[4]       ;// = {*(matrices + 8),  *(matrices + 9),  *(matrices + 10), *(matrices + 11)};
  float4 const* normalmatrix                    = matrices + 12;// = {*(matrices + 12), *(matrices + 13), *(matrices + 14), *(matrices + 15)};
  float4 const* modelviewprojectionmatrixinverse= matrices + 16;// = {*(matrices + 16), *(matrices + 17), *(matrices + 18), *(matrices + 19)};

  /********************************************************************************
  * second: if there are fragments - sort them by depth
  ********************************************************************************/
  bubble_sort_indexlist_1 ( fragindex, nfragments, indexlist );
  __syncthreads();

  if ( !detect_face_by_sampling )
  {
    /********************************************************************************
    * raycast for surface intersections and update lists
    ********************************************************************************/
    compute_face_intersections_per_fragment ( fragindex, 
                                              nfragments,
                                              nface_intersections,
                                              nface_intersection_tests,
                                              coords,
                                              int2_t(width, height),
                                              surface_transparency, 
                                              fixed_newton_epsilon,
                                              max_iterations_newton,
                                              nearplane,
                                              farplane,
                                              indexlist, 
                                              surfacedatabuffer,
                                              surfacepointsbuffer,
                                              modelviewmatrix,
                                              modelviewmatrixinverse,
                                              modelviewprojectionmatrixinverse);
  } else {

    merge_intervals         ( fragindex, indexlist );
    //__syncthreads();

    /********************************************************************************
    * raycast for surface intersections and update lists
    ********************************************************************************/
    int2 tileid = int2_t ( coords.x % tilesize.x, coords.y % tilesize.y );

    compute_face_intersections_per_domain_intersection ( fragindex, 
                                                          nfragments,
                                                          nface_intersections,
                                                          nface_intersection_tests,
                                                          npoint_evaluations,
                                                          coords,
                                                          int2_t(width, height),
                                                          surface_transparency, 
                                                          fixed_newton_epsilon,
                                                          max_iterations_newton,
                                                          detect_implicit_inflection,
                                                          detect_implicit_extremum,
                                                          nearplane,
                                                          farplane,
                                                          min_sample_distance,
                                                          max_sample_distance,
                                                          adaptive_sample_scale,
                                                          indexlist,
                                                          allocation_grid,
                                                          tileid,
                                                          tilesize,
                                                          pagesize,
                                                          surfacedatabuffer,
                                                          surfacepointsbuffer,
                                                          volumedatabuffer,
                                                          volumepointsbuffer,
                                                          modelviewmatrix,
                                                          modelviewmatrixinverse,
                                                          modelviewprojectionmatrixinverse);
  }

  /********************************************************************************
  * sort fragment list after surface intersections
  ********************************************************************************/ 
  __syncthreads();
  bubble_sort_indexlist_1 ( fragindex, nfragments, indexlist );
  __syncthreads();

  /********************************************************************************
  * start traversing intersections for isosurface intersections and shading
  ********************************************************************************/ 
  fragment current_fragment (fragindex, indexlist);
  fragment last_fragment;

  face_info current_face;
  face_info last_face;
  ray_state ray;
  unsigned processed_fragments = 0;

  /********************************************************************************
  * traverse depth-sorted fragments
  ********************************************************************************/ 
  //while ( !ray.abort && ++processed_fragments < nfragments )
  for ( processed_fragments = 0; processed_fragments < nfragments && !ray.abort; ++processed_fragments )
  {
    /********************************************************************************
    * ray setup
    ********************************************************************************/ 
    if ( !ray.initialized )
    {
      ray.initialized = true;
      ray.initialize ( coords, width, height, 
                        current_fragment.depth, 
                        modelviewprojectionmatrixinverse, 
                        modelviewmatrixinverse );
    }

    /********************************************************************************
    * determine successive faces
    ********************************************************************************/ 
    current_face          = face_info ( surfacedatabuffer, volumedatabuffer, current_fragment );
    current_depth         = current_fragment.depth;

    bool faces_have_same_id    = current_face.surface_unique_id == last_face.surface_unique_id;
    bool faces_have_same_depth = fabs(current_face.depth - last_face.depth) < fixed_newton_epsilon;
    bool equals_last_face      = faces_have_same_id && faces_have_same_depth;

    /********************************************************************************
    * retrieve volume enclosed by fragment
    ********************************************************************************/
    if ( current_face.is_surface && 
          last_face.is_surface &&
          !equals_last_face )
    {
      unsigned contained_volume_id = determine_contained_volume ( last_face, current_face );

      volume_info volume;
      volume.load_from_volume_id ( volumedatabuffer, attributedatabuffer, contained_volume_id );

      float attribute_min = attributedatabuffer[volume.attribute_data_id].x;
      float attribute_max = attributedatabuffer[volume.attribute_data_id].y;

      bool contains_iso_value = is_in_range ( iso_threshold, attribute_min, attribute_max );

      if ( contained_volume_id != 0 && contains_iso_value )
      {
#if SEARCH_FOR_ISOSURFACE
        bool continue_isosurface_search                   = true;
        int isosurface_intersections_per_volume           = 0;
        int const max_isosurface_intersections_per_volume = 3;

        float4 ray_entry = screen_to_object_coordinates (coords, int2_t(width, height), last_face.depth, modelviewprojectionmatrixinverse);
        float4 ray_exit  = screen_to_object_coordinates (coords, int2_t(width, height), current_face.depth, modelviewprojectionmatrixinverse);
        float3 uvw_guess = last_fragment.uvw(surfacedatabuffer, contained_volume_id);

        sample current_sample, iso_sample;

        current_sample.volume = volume;
        current_sample.uvw    = uvw_guess;
        current_sample.p      = ray_entry;

        while ( continue_isosurface_search && 
                isosurface_intersections_per_volume < max_isosurface_intersections_per_volume )
        {
          ++isosurface_intersections_per_volume;

          continue_isosurface_search = search_volume_for_iso_surface ( volumepointsbuffer,
                                                                        attributepointsbuffer,
                                                                        volume.volume_points_id,
                                                                        volume.attribute_points_id,
                                                                        current_sample,
                                                                        current_sample,
                                                                        iso_sample,
                                                                        ray,
                                                                        ray_exit,
                                                                        iso_threshold,
                                                                        adaptive_sampling,
                                                                        min_sample_distance,
                                                                        max_sample_distance,
                                                                        adaptive_sample_scale,
                                                                        true,
                                                                        fixed_newton_epsilon,
                                                                        max_iterations_newton,
                                                                        max_steps_binary_search,
                                                                        npoint_evaluations );

          if ( continue_isosurface_search ) 
          {
            float4 iso_color = iso_sample.shade_isosurface ( modelviewmatrix, normalmatrix, global_attribute_min, global_attribute_max );
            ray.blend ( float4_t( iso_color.x, iso_color.y, iso_color.z, isosurface_transparency ));
          }

          /********************************************************************************
          * early ray termination
          ********************************************************************************/ 
          if ( ray.out_opacity < 0.01f )
          {
            break;
          }
        }
#endif
      }
    }

#if SHADE_OUTER_SURFACE
    /********************************************************************************
    * shade entry face
    ********************************************************************************/ 
    if ( current_face.is_surface && 
          current_face.is_outer &&
          !equals_last_face
        ) 
    {
      sample face_sample;
      face_sample.volume.load_from_volume_id ( volumedatabuffer, attributedatabuffer, current_face.volume_data_id );
      face_sample.uvw = current_fragment.compute_uvw ( surfacedatabuffer, current_fragment.uv );

      horner_volume_derivatives<float2, 1> ( attributepointsbuffer, 
                                              face_sample.volume.attribute_points_id, 
                                              face_sample.volume.volume_order, 
                                              face_sample.uvw, 
                                              face_sample.a, 
                                              face_sample.da_du, 
                                              face_sample.da_dv, 
                                              face_sample.da_dw );  

      //float4 attrib_color = float4_t(1.0, 1.0, 1.0, 1.0);
      float diffuse         = current_face.shade(current_fragment.uv, surfacepointsbuffer, modelviewmatrix, normalmatrix);
      float relative_attrib = normalize ( face_sample.a.x, global_attribute_min, global_attribute_max );
      float4 relative_color = diffuse * transferfunction ( relative_attrib ); 
      relative_color.w      = surface_transparency;
       
      ray.blend ( relative_color );
    }

#endif

    /********************************************************************************
    * store this passes information
    ********************************************************************************/ 
    if ( current_face.is_surface ) 
    {
      last_face = current_face;
      last_fragment = current_fragment;
    }
      
    if ( !current_fragment.has_next() )
    {
      ray.abort = true;
    } else {
      current_fragment = current_fragment.get_next(indexlist);
    }

    /********************************************************************************
    * early ray termination
    ********************************************************************************/ 
    if ( ray.out_opacity < 0.01f )
    {
      break;
    }
  } // for all fragments

  /********************************************************************************
  * volume ray casting ended -> check if there is an external hit behind current ray position
  ********************************************************************************/   
  if ( current_depth <= external_depth && external_depth < 1.0f ) 
  {
    current_depth = external_depth;
    ray.out_color = ray.out_color + ray.out_opacity * external_color;
  }

  /*
  if ( show_face_intersection_tests ) {
    ray.out_color = transferfunction(float(nface_intersection_tests)/16.0f);
  }
 
  if ( show_face_intersections ) {
    ray.out_color = transferfunction(float(npoint_evaluations)/64.0);
  } */

  surf2Dwrite ( current_depth,  depthtexture, coords.x*sizeof(float), coords.y );
  surf2Dwrite ( ray.out_color,  colortexture, coords.x*sizeof(float4), coords.y );
}

#endif
