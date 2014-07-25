/********************************************************************************
*
* Copyright (C) 2009-2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : sample_fragmentlists.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_SAMPLE_FRAGMENTLISTS_H
#define GPUCAST_SAMPLE_FRAGMENTLISTS_H

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

#include <isosurface/compute_sampling_position.h>
#include <isosurface/ray_state.h>
#include <isosurface/fragment.h>
#include <isosurface/fragment_range.h>
#include <isosurface/sample.h>
#include <isosurface/face_info.h>
#include <isosurface/history.h>
#include <isosurface/target_function.h>
#include <isosurface/binary_search_for_isosurface.h>
#include <isosurface/binary_search_for_face.h>

#include <fragmentlist/bubble_sort.h>
#include <fragmentlist/verify_depth_order.h>
#include <fragmentlist/determine_next_fragment.h>
#include <fragmentlist/count_fragments.h>

#define SHOW_NUMBER_OF_SAMPLES 1
#define HISTORY_USING_LAST_VOLUME_ONLY 0

///////////////////////////////////////////////////////////////////////////////
struct intersection
{
  __device__ inline intersection () 
  {}

  __device__ inline intersection ( float4 const& c, float d )
    : color (c),
      depth (d)
  {}

  float4 color;
  float  depth;
};


///////////////////////////////////////////////////////////////////////////////
struct intersection_stack
{
  intersection* intersections;
  unsigned      size;

  __device__ inline
  intersection_stack ( intersection* shared_mem )
    : intersections (shared_mem),
      size (0)
  {}

  __device__ inline void
  clear () 
  {
    size = 0;
  }

  __device__ inline void
  push ( intersection const& i )
  {
    intersections[size] = i;
    ++size;
  }

  __device__ inline void
  sort () const
  {
    int n = size;
    do {
      int newn = 1;
      for ( int i = 0; i < n-1; ++i ) 
      {
        if (intersections[i].depth > intersections[i+1].depth)
        {
          swap(i, i+1);
          newn = i+1;
        } // ende if
      } // ende for
      n = newn;
    } while (n > 1);
  }

  __device__ inline void 
  swap ( int lhs_id, int rhs_id ) const 
  {
    intersection tmp = intersections[lhs_id];
    intersections[lhs_id] = intersections[rhs_id];
    intersections[rhs_id] = tmp;
  }

  __device__ inline void 
  make_unique ( float epsilon = 0.00001f )
  {
    float    current_depth = 0;
    unsigned unique_index  = 0;
    for ( int i = 0; i != size; ++i ) 
    {
      if ( fabs(current_depth - intersections[i].depth) > epsilon )
      {
        if ( i != unique_index )
        {
          intersections[unique_index] = intersections[i];
        }
        ++unique_index;
      } 
      current_depth = intersections[i].depth;
    }
    size = unique_index;
  }

  __device__ inline void
  blend ( ray_state& ray, float depth )
  {
    int blended = 0;
    for ( int i = 0; i != size; ++i ) 
    {
      if ( intersections[i].depth < depth ) 
      {
        ray.blend(intersections[i].color);
        ++blended;
      } else {
        break;
      }
    }

    if ( blended > 0 ) 
    {
      int j = 0;
      for ( int i = blended; i != size; ++i, ++j )
      {
        intersections[j] = intersections[i];
      }
      size -= blended;
    }
  }

};



///////////////////////////////////////////////////////////////////////////////
__device__ inline void
get_next_fragment_range ( int             start_index,
                          uint4 const*    surfacedatabuffer,
                          uint4 const*    fragmentbuffer,
#if HISTORY_USING_LAST_VOLUME_ONLY
                          unsigned        last_volume_index,
#else
                          history const&  history,
#endif
                          fragment_range& range )
{
  unsigned current_index = start_index;

  range.clear();  

  while ( current_index != 0 )
  {
    fragment f ( current_index, fragmentbuffer );

    unsigned vid0 = 0;
    unsigned vid1 = 0;
    f.get_volume_ids ( surfacedatabuffer, vid0, vid1 );

    // continue with existing range
    if ( range.valid() ) 
    {
      // adjacent volume matches current range
      if ( range.get_volume_id() == vid0 || 
           range.get_volume_id() == vid1 )
      {
        range.add(f);
      }

    // initialize new range
    } else {
      // if volume 0 wasn't processed already -> start with volume 0
#if HISTORY_USING_LAST_VOLUME_ONLY
      if ( last_volume_index != vid0 && vid0!=0 ) {
#else
      if ( !history.contains(vid0) && vid0!=0 ) {
#endif
        range.initialize(f, vid0);
      } else { // if volume 1 wasn't processed already -> start with volume 1
#if HISTORY_USING_LAST_VOLUME_ONLY
       if ( last_volume_index != vid1 && vid0!=1 ) {
#else
        if ( !history.contains(vid1) && vid1!=0 ) {
#endif
          range.initialize(f, vid1);
        } else {
          // both volumes have been processed -> go to next fragment
        }
      }
    }
          
    // go to next fragment
    current_index = f.next;
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned
determine_adjacent_face_offset ( float3 const& uvw0, float3 const& uvw1 )
{
  unsigned max_u = unsigned((uvw0.x - 1.0f) * (uvw1.x - 1.0f) < 0.0);
  unsigned min_u = unsigned((uvw0.x       ) * (uvw1.x       ) < 0.0);
  unsigned max_v = unsigned((uvw0.y - 1.0f) * (uvw1.y - 1.0f) < 0.0);
  unsigned min_v = unsigned((uvw0.y       ) * (uvw1.y       ) < 0.0);
  unsigned max_w = unsigned((uvw0.z - 1.0f) * (uvw1.z - 1.0f) < 0.0);
  unsigned min_w = unsigned((uvw0.z       ) * (uvw1.z       ) < 0.0);

  return (1 - min_w + max_w) * 9 +
         (1 - min_v + max_v) * 3 +
         (1 - min_u + max_u);
}


///////////////////////////////////////////////////////////////////////////////
__device__
void sample_fragmentlists  ( int                      threadidx,
                             int                      width, 
                             int                      height,
                             int2 const&              coords,
                             float                    nearplane,
                             float                    farplane,
                             float3 const&            background,
                             int2 const&              tilesize,
                             unsigned                 pagesize,
                             unsigned                 volume_info_offset,
                             float                    iso_threshold,
                             int                      show_isosides,
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
                             float4 const*            matrices,
                             uint4*                   indexlist,
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
  * shared memory allocation
  ********************************************************************************/
  /*__shared__ */ fragment      shared_mem_fragment_buffer[12];
  /*__shared__ */ unsigned      shared_mem_history[32];
  /*__shared__ */ intersection  shared_mem_intersection_stack[12];

#if HISTORY_USING_LAST_VOLUME_ONLY
  unsigned            last_volume_index = 0;
#else
  history             volume_history (shared_mem_history);
#endif
  fragment_range      current_range (shared_mem_fragment_buffer);
  intersection_stack  color_stack (shared_mem_intersection_stack);

  /********************************************************************************
  * read header pointer
  ********************************************************************************/
  int fragindex = 0;
  surf2Dread ( &fragindex, headpointer_image, coords.x*sizeof(float), coords.y );

  unsigned nfragments = count_fragments(indexlist, fragindex);
  bubble_sort_indexlist_1 ( fragindex, nfragments, indexlist );

  /********************************************************************************
  * read external color and depth
  ********************************************************************************/
  float4 external_color;
  surf2Dread ( &external_color, external_color_depth_image, coords.x*sizeof(float4), coords.y );
  float external_depth = external_color.w;
  external_color.w = 1.0f;

  /********************************************************************************
  * allocate memory for matrices
  ********************************************************************************/  
  float4 const* modelviewmatrix                   = matrices;
  float4 const* modelviewmatrixinverse            = matrices + 4;
  float4 const* modelviewprojectionmatrix         = matrices + 8;       
  float4 const* normalmatrix                      = matrices + 12;                   
  float4 const* modelviewprojectionmatrixinverse  = matrices + 16;

  /********************************************************************************
  * init samples and ray setup
  ********************************************************************************/ 
  sample    current_sample, last_sample;
  ray_state ray;

  /********************************************************************************
  * start ray casting
  ********************************************************************************/
  if ( nfragments <= 0 )
  {
    if ( external_depth < 1.0f ) 
    {
      surf2Dwrite ( external_depth, depthtexture, coords.x*sizeof(float), coords.y );
      surf2Dwrite ( external_color, colortexture, coords.x*sizeof(float4), coords.y );
    }
  } else {
     // set depth to zero
    ray.out_depth = 0.0f;
    int      max_fragment_ranges = 128;
    unsigned processed_ranges    = 0;
    unsigned processed_samples   = 0;

    unsigned tmp_found_faces = 0;

    while ( !ray.abort ) 
    {
      //////////////////////////////////////////////////////
      // check for crititcal abort after max processed ranges
      //////////////////////////////////////////////////////
      if ( processed_ranges++ > max_fragment_ranges ) 
      { 
        ray.out_color = float4_t(float(nfragments)/8.0f, 0.0, 0.0, 1.0);
        ray.abort     = true;
        break;
      }

      //////////////////////////////////////////////////////
      // retrieve next range from fragment list
      //////////////////////////////////////////////////////
      current_range.clear();

#if HISTORY_USING_LAST_VOLUME_ONLY
      get_next_fragment_range(fragindex, surfacedatabuffer, indexlist, last_volume_index, current_range);
#else
      get_next_fragment_range(fragindex, surfacedatabuffer, indexlist, volume_history, current_range);
#endif

      color_stack.sort();
      color_stack.make_unique(fixed_newton_epsilon);
      float next_range_entry_depth = current_range.valid() ? current_range.first_fragment().depth : 1.0f;
      color_stack.blend(ray, next_range_entry_depth);

      if ( ray.out_opacity < 0.01 )
      {
        break;
      }

      //////////////////////////////////////////////////////
      // could not get a valid range
      //////////////////////////////////////////////////////
      if ( !current_range.valid() ) {
        ray.abort = true;
      } else {
        //////////////////////////////////////////////////////
        // process found fragment range
        //////////////////////////////////////////////////////

        // initialize ray if not already initialized
        if ( !ray.initialized ) 
        {
          ray.initialized = true;
          ray.initialize ( coords, width, height, 
                           current_range.first_fragment().depth, 
                           modelviewprojectionmatrixinverse, 
                           modelviewmatrixinverse );
        }

        //////////////////////////////////////////////////////
        // initialize sample bounds
        //////////////////////////////////////////////////////
        float4 entry_point           = screen_to_object_coordinates ( coords, int2_t(width, height), current_range.first_fragment().depth, modelviewprojectionmatrixinverse );
        float4 exit_point            = screen_to_object_coordinates ( coords, int2_t(width, height), current_range.last_fragment().depth, modelviewprojectionmatrixinverse );
        float distance_entry_to_exit = length(exit_point - entry_point);

        current_sample.volume.load_from_volume_id ( volumedatabuffer, attributedatabuffer, current_range.get_volume_id() );
        current_sample.p              = entry_point;
        last_sample                   = current_sample;

        int samples = 0;
        unsigned max_samples_in_range = 2 + ceil(distance_entry_to_exit / (min_sample_distance * current_sample.volume.volume_bbox_size));

        bool processing_first_sample  = true;
        bool processing_last_sample   = false;

        /////////////////////////////////////////////////////////////////////
        // sample along all fragments that belong to the volume
        /////////////////////////////////////////////////////////////////////

        while ( !ray.abort && 
                !current_range.abort() && 
                processed_samples < max_samples_in_range )
        {
          ++processed_samples;

          float3 uvw_guess;
          float4 sampling_pos;

          /////////////////////////////////////////////////////////////////////
          // 1. compute sampling position and initial guess
          /////////////////////////////////////////////////////////////////////

          // processing first sample of range
          if ( processing_first_sample )
          {
            sampling_pos            = entry_point;
            uvw_guess               = current_range.first_fragment().uvw ( surfacedatabuffer, current_range.get_volume_id() );
          // sample in between range
          } else { 
            if ( last_sample.inversion_success ) 
            {
              // last sample was valid -> use for adaptive sampling
              uvw_guess    = last_sample.uvw;
              sampling_pos = compute_sampling_position_adaptively ( last_sample.p,
                                                                    float4_t(ray.direction.x, ray.direction.y, ray.direction.z, 0.0f),
                                                                    last_sample.volume.volume_bbox_size,
                                                                    min_sample_distance,
                                                                    max_sample_distance,
                                                                    adaptive_sample_scale,
                                                                    iso_threshold,
                                                                    last_sample, 
#if 0
                                                                    false,
#else
                                                                    last_sample.volume.outer_facemask > 0,
                                                                    
#endif
                                                                    current_sample.volume.outer_facemask
                                                                   );
            } else {
              // last sample was not valid -> static sampling
              sampling_pos             = last_sample.p + min_sample_distance * last_sample.volume.volume_bbox_size * float4_t(ray.direction.x, ray.direction.y, ray.direction.z, 0.0f);
              float sampling_pos_depth = compute_depth_from_object_coordinates (modelviewmatrix, sampling_pos, nearplane, farplane);
              uvw_guess                = current_range.interpolate_guess ( sampling_pos_depth, surfacedatabuffer );
            }
          }

          // if sampling position is behind exit point -> clamp to exit point and use uvw guess from exit
          if ( length ( sampling_pos - entry_point ) > distance_entry_to_exit )
          {
            current_range.abort(true);
            sampling_pos           = exit_point; // clamp to range end and stop sampling for this volume
            uvw_guess              = current_range.last_fragment().uvw ( surfacedatabuffer, current_range.get_volume_id() );
            processing_last_sample = true;
          }

          /////////////////////////////////////////////////////////////////////
          // 2. try to solve inverse mapping
          /////////////////////////////////////////////////////////////////////
          current_sample.inversion_success = newton_volume_unbound ( volumepointsbuffer, 
                                                                     current_sample.volume.volume_points_id, 
                                                                     current_sample.volume.volume_order,
                                                                     uvw_guess,
                                                                     current_sample.uvw,
                                                                     sampling_pos,
                                                                     current_sample.p,
                                                                     current_sample.dp_du,
                                                                     current_sample.dp_dv,
                                                                     current_sample.dp_dw,
                                                                     ray.n1,
                                                                     ray.n2,
                                                                     ray.direction,
                                                                     ray.d1,
                                                                     ray.d2,
                                                                     fixed_newton_epsilon,
                                                                     max_iterations_newton );

          // first and last samples are outside because they lie on convex hull
          current_sample.is_inside = current_sample.uvw_in_domain() && 
                                     !processing_first_sample &&
                                     !processing_last_sample;

          if ( !current_sample.inversion_success ) // inversion for current sample failed
          {
             // reset current sampling position and uvw, because iteration might have diverged
            current_sample.p   = sampling_pos;
            current_sample.uvw = uvw_guess;
          } else { // inversion for current sample succeed

            // evaluate attribute function for uvw
            horner_volume_derivatives<float2, 1> ( attributepointsbuffer, 
                                                   current_sample.volume.attribute_points_id, 
                                                   current_sample.volume.volume_order, 
                                                   current_sample.uvw,
                                                   current_sample.a,
                                                   current_sample.da_du,
                                                   current_sample.da_dv,
                                                   current_sample.da_dw );

            /////////////////////////////////////////////////////////////////////
            // make sure first and last fragment are out of domain!
            /////////////////////////////////////////////////////////////////////
            if ( processing_first_sample || processing_last_sample ) 
            {
//#define SET_DOMAIN_POINT_OUT_OF_DOMAIN_BY_FACE_TYPE
#ifdef SET_DOMAIN_POINT_OUT_OF_DOMAIN_BY_FACE_TYPE
              unsigned face_type;
              if ( processing_last_sample )
                face_type = surfacedatabuffer[current_range.last_fragment().surface_data_id+2].w;
              if ( processing_first_sample ) 
                face_type = surfacedatabuffer[current_range.first_fragment().surface_data_id+2].w;

              current_sample.uvw = clamp ( current_sample.uvw, float3_t(0.0, 0.0, 0.0), float3_t(1.0, 1.0, 1.0));
              if ( face_type == 0 ) current_sample.uvw.x = -fixed_newton_epsilon;
              if ( face_type == 1 ) current_sample.uvw.x = 1.0f + fixed_newton_epsilon;
              if ( face_type == 2 ) current_sample.uvw.y = -fixed_newton_epsilon;
              if ( face_type == 3 ) current_sample.uvw.y = 1.0f + fixed_newton_epsilon;
              if ( face_type == 4 ) current_sample.uvw.z = -fixed_newton_epsilon;
              if ( face_type == 5 ) current_sample.uvw.z = 1.0f + fixed_newton_epsilon; 
#else
              if ( current_sample.uvw.x <= 0.0f + 10.0f * fixed_newton_epsilon ) current_sample.uvw.x = -fixed_newton_epsilon;
              if ( current_sample.uvw.x >= 1.0f - 10.0f * fixed_newton_epsilon ) current_sample.uvw.x = 1.0 + fixed_newton_epsilon;
              if ( current_sample.uvw.y <= 0.0f + 10.0f * fixed_newton_epsilon ) current_sample.uvw.y = -fixed_newton_epsilon;
              if ( current_sample.uvw.y >= 1.0f - 10.0f * fixed_newton_epsilon ) current_sample.uvw.y = 1.0 + fixed_newton_epsilon;
              if ( current_sample.uvw.z <= 0.0f + 10.0f * fixed_newton_epsilon ) current_sample.uvw.z = -fixed_newton_epsilon;
              if ( current_sample.uvw.z >= 1.0f - 10.0f * fixed_newton_epsilon ) current_sample.uvw.z = 1.0 + fixed_newton_epsilon;
#endif
            }

            /////////////////////////////////////////////////////////////////////
            // 3. found to successive samples for which the inversion succeeded
            /////////////////////////////////////////////////////////////////////
            if ( last_sample.inversion_success && 
                 current_sample.inversion_success ) 
            {
              /////////////////////////////////////////////////////////////////////
              // 3.1 check for in/outside change
              /////////////////////////////////////////////////////////////////////
              sample face_sample     = current_sample;

//#define SEARCH_EXACT_FACE_INTERSECTION_BY_BISECTION_VOLUME_SAMPLES
#ifdef SEARCH_EXACT_FACE_INTERSECTION_BY_BISECTION_VOLUME_SAMPLES
              bool face_intersection = current_sample.is_inside != last_sample.is_inside;
              
              if ( face_intersection )
              {
                float out_face_depth;
                if ( !current_sample.is_inside ) {
                  out_face_depth = compute_depth_from_object_coordinates ( modelviewmatrix, current_sample.p, nearplane, farplane );
                } else {
                  out_face_depth = compute_depth_from_object_coordinates ( modelviewmatrix, last_sample.p, nearplane, farplane );
                }

                unsigned  face_type       = current_sample.compute_face_type(last_sample);
                unsigned  surface_data_id = current_range.get_surface_data_id ( surfacedatabuffer, out_face_depth, face_type);
                bool      is_outer_face   = surfacedatabuffer[surface_data_id+1].z == 0 && surface_data_id != 0;

                if ( is_outer_face )
                {  
                  tmp_found_faces++;
                  binary_search_for_face (  volumepointsbuffer, 
                                            attributepointsbuffer, 
                                            current_sample.volume.volume_points_id, 
                                            current_sample.volume.attribute_points_id, 
                                            last_sample,
                                            current_sample,
                                            face_sample,
                                            ray,
                                            iso_threshold,
                                            fixed_newton_epsilon,
                                            max_iterations_newton, 
                                            max_steps_binary_search );

                  out_face_depth    = compute_depth_from_object_coordinates ( modelviewmatrix, face_sample.p, nearplane, farplane );
                  float4 face_color = face_sample.shade_face (modelviewmatrix, normalmatrix, face_type, global_attribute_min, global_attribute_max);
                  face_color.w      = surface_transparency;

                  intersection face_hit ( face_color, out_face_depth );
                  color_stack.push (face_hit);
                }
              }
#else
              /////////////////////////////////////////////////////////////////////
              // 3.1 check if samples changed sides of domain
              /////////////////////////////////////////////////////////////////////
              isoparametric_transition transition;
              current_sample.compute_transition(last_sample, transition);

              if ( transition.intersects_boundary() )
              {
                float out_face_depth;
                if ( !current_sample.is_inside ) {
                  out_face_depth = compute_depth_from_object_coordinates ( modelviewmatrix, current_sample.p, nearplane, farplane );
                } else {
                  out_face_depth = compute_depth_from_object_coordinates ( modelviewmatrix, last_sample.p, nearplane, farplane );
                }

                for ( int face_type = 0; face_type != 6; ++face_type )
                {
                  if ( transition.hit[face_type] )
                  {
                    
                    unsigned  surface_data_id = 0;
                    float const* volume_data_ptr = &volumedatabuffer[current_range.get_volume_id()+6].x;
                    surface_data_id = floatBitsToInt(*(volume_data_ptr + face_type));
                    
                    bool      is_outer_face   = ((current_sample.volume.outer_facemask >> face_type) & 0x0001); //surfacedatabuffer[surface_data_id+1].z == 0 && surface_data_id != 0;

                    if ( is_outer_face && surface_data_id != 0 ) 
                    {
                      uint4    surface_order_info = surfacedatabuffer[surface_data_id + 2];
                      unsigned surface_points_id  = surfacedatabuffer[surface_data_id + 1].y;
                      uint2    face_order         = uint2_t ( surface_order_info.y, surface_order_info.z );
                      unsigned tmp                = 0;

                      bool face_intersect = newton_surface ( surfacepointsbuffer, 
                                                             surface_points_id, 
                                                             transition.uv[face_type], 
                                                             fixed_newton_epsilon, 
                                                             max_iterations_newton, 
                                                             face_order, 
                                                             ray.n1,
                                                             ray.n2,
                                                             ray.d1,
                                                             ray.d2,
                                                             face_sample.p,
                                                             face_sample.dp_du,
                                                             face_sample.dp_dv,
                                                             tmp );

                      if ( face_intersect )
                      {
                        if ( face_type == 0 || face_type == 1 ) face_sample.uvw = float3_t ( face_type%2, transition.uv[face_type].x, transition.uv[face_type].y );
                        if ( face_type == 2 || face_type == 3 ) face_sample.uvw = float3_t ( transition.uv[face_type].x, face_type%2, transition.uv[face_type].y );
                        if ( face_type == 4 || face_type == 5 ) face_sample.uvw = float3_t ( transition.uv[face_type].x, transition.uv[face_type].y, face_type%2 );
                        
                        // evaluate attribute function for uvw
                        horner_volume_derivatives<float2, 1> ( attributepointsbuffer, 
                                                               face_sample.volume.attribute_points_id, 
                                                               face_sample.volume.volume_order, 
                                                               face_sample.uvw,
                                                               face_sample.a,
                                                               face_sample.da_du,
                                                               face_sample.da_dv,
                                                               face_sample.da_dw );

                        out_face_depth     = compute_depth_from_object_coordinates ( modelviewmatrix, face_sample.p, nearplane, farplane );
                        float4 face_normal = cross ( face_sample.dp_du, face_sample.dp_dv );
                        float4 face_color  = face_sample.shade (modelviewmatrix, normalmatrix, face_normal, global_attribute_min, global_attribute_max);
                        face_color.w       = surface_transparency;

                        intersection face_hit ( face_color, out_face_depth );
                        color_stack.push (face_hit);
                      } // if boundary was intersected by newton iteration
                    } // if this boundary is outer face
                  } // if this boundary is intersected
                } // for each potential boundary intersection
              } // intersects one or more boundaries
#endif

              /////////////////////////////////////////////////////////////////////
              // check for sign change and binary search for isosurface
              /////////////////////////////////////////////////////////////////////
              if ( ( current_sample.a.x - iso_threshold ) * ( last_sample.a.x - iso_threshold ) < 0.0f ) 
              {
                sample iso_sample = current_sample;
                binary_search_for_isosurface ( volumepointsbuffer, 
                                               attributepointsbuffer, 
                                               current_sample.volume.volume_points_id, 
                                               current_sample.volume.attribute_points_id, 
                                               last_sample,
                                               current_sample,
                                               iso_sample,
                                               ray,
                                               iso_threshold,
                                               fixed_newton_epsilon,
                                               max_iterations_newton, 
                                               max_steps_binary_search,
                                               processed_samples );

                // blend found isosurface intersection
                if ( iso_sample.inversion_success && iso_sample.uvw_in_domain() )
                {
                  //ray.blend_isosurface ( iso_sample, isosurface_transparency, modelviewmatrix, normalmatrix, global_attribute_min, global_attribute_max);
                  float isosurface_depth  = compute_depth_from_object_coordinates ( modelviewmatrix, iso_sample.p, nearplane, farplane );

                  float4 isosurface_color = iso_sample.shade_isosurface (modelviewmatrix, normalmatrix, global_attribute_min, global_attribute_max);
                  isosurface_color.w      = isosurface_transparency;

                  intersection isosurface_hit ( isosurface_color, isosurface_depth );
                  color_stack.push (isosurface_hit);
                }

              } // found sign change of attribute function
            } else { 
              // inversion for last sample was not successful -> just continue sampling
            }
          }

          // store current sample as last sample for next iteration
          last_sample = current_sample;

          if ( processing_first_sample ) processing_first_sample = false;

        } // sampling current fragment range

        // push processed volume into history
#if HISTORY_USING_LAST_VOLUME_ONLY
        last_volume_index = current_range.get_volume_id();
#else
        volume_history.push_back(current_range.get_volume_id());
#endif
        
      } // found fragment range
    } // found fragment

    // write computed color to image
    ray.out_depth = 0.3f;

#if SHOW_NUMBER_OF_SAMPLES
    if ( max_steps_binary_search == 15 )
    {
      if ( length(ray.out_color) < 1.0 )
        ray.out_color = transferfunction(float(processed_samples)/256.0);
    }
#endif

    surf2Dwrite ( ray.out_depth, depthtexture, coords.x*sizeof(float), coords.y );
    surf2Dwrite ( ray.out_color, colortexture, coords.x*sizeof(float4), coords.y );
  }
}

#endif
