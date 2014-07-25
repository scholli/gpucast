#ifndef LIBGPUCAST_SEARCH_VOLUME_FOR_ISO_SURFACE_H
#define LIBGPUCAST_SEARCH_VOLUME_FOR_ISO_SURFACE_H

#include "./local_memory_config.h"

#include "math/horner_volume.h"
#include "math/newton_volume.h"
#include "math/ray_to_plane_intersection.h"
#include "math/point_ray_distance.h"

#include "isosurface/binary_search_for_isosurface.h"
#include "isosurface/newton_search_for_isosurface.h"
#include "isosurface/secant_search_for_isosurface.h"
#include "isosurface/compute_sampling_position.h"
#include "isosurface/compute_iso_normal.h"
#include "isosurface/target_function.h"
#include "isosurface/validate_bezier_domain.h"
#include "isosurface/validate_isosurface_intersection.h"

__device__ inline
bool search_volume_for_iso_surface ( 
                                     float4 const*    volumepointsbuffer,
                                     float2 const*    attributepointsbuffer,
                                     int              point_base_id,
                                     int              attribute_base_id,
                                     uint3 const&     order,
                                     float3 const&    start_uvw,
                                     float            iso_threshold,
                                     float4 const&    ray_entry,
                                     float4 const&    ray_exit,
                                     bool             adaptive_sampling,
                                     float            volume_bbox_diameter,
                                     float            min_sample_distance,
                                     float            max_sample_distance,
                                     float            adaptive_sample_scale,
                                     bool             adaptive_newton,
                                     float            fixed_newton_epsilon,
                                     unsigned         max_iterations_newton,
                                     unsigned         max_steps_binary_search,
                                     float4*          iso_hit_position,
                                     float2*          iso_hit_attribute,
                                     float4*          iso_hit_normal,
                                     float3*          iso_hit_uvw,
                                     float4*          next_ray_entry,
                                     float3*          next_uvw,
                                     unsigned&        total_samples )
{
  // create ray in object coordinates
  float4 ray_direction   = ray_exit - ray_entry;
  ray_direction.w        = 0.0f;
  float  sample_distance = length(ray_direction);

  int    max_samples     = (int)( ceil ( sample_distance / ( min_sample_distance * volume_bbox_diameter ) ) );
  max_samples            = max ( 2, max_samples ); // use at least first entry and exit point

  // normalize ray direction
  ray_direction          = ray_direction / sample_distance;  

  // convert ray to intersection of two planes
  float3 n1   = float3_t(0.0f, 0.0f, 0.0f);
  float3 n2   = float3_t(0.0f, 0.0f, 0.0f);
  float  d1   = 0.0f;
  float  d2   = 0.0f;

  ray_to_plane_intersection ( ray_entry, ray_direction, &n1, &n2, &d1, &d2 );

  // initialize sampling by allocating variables and evaluating volume
  float4 point,  pdu, pdv, pdw;
  float2 attrib, adu, adv, adw;
  float3 uvw_current           = start_uvw; 
  float3 uvw_next              = float3_t(0.0f, 0.0f, 0.0f);
  float4 sample_position       = ray_entry;
  bool   sample_end_reached    = false;
  bool   last_sample_valid     = false;
  float2 last_sample_attribute;
  float4 last_sample_position  = float4_t(0.0f, 0.0f, 0.0f, 0.0f);

  /********************************************************************************
  * search fragment interval for iso surface
  ********************************************************************************/  
  for ( int i = 0; i != max_samples; ++i )
  {
    ++total_samples;

    float newton_epsilon = 0.0f;

    if ( adaptive_newton )
    {
    } else {
      newton_epsilon  = fixed_newton_epsilon;
    }

    newton_volume(  volumepointsbuffer, 
                    point_base_id,
                    order,
                    uvw_current,
                    uvw_next,
                    sample_position,
                    point,
                    pdu,
                    pdv,
                    pdw,
                    n1,
                    n2,
                    float3_t(ray_direction.x, ray_direction.y, ray_direction.z),
                    d1,
                    d2,
                    newton_epsilon,
                    max_iterations_newton );

    //horner_volume_derivatives ( pointbuffer,     point_base_id,     order, uvw_next, &point,  &pdu, &pdv, &pdw );  
    horner_volume_derivatives<float2, 1> ( attributepointsbuffer, attribute_base_id, order, uvw_next, attrib, adu, adv, adw );  

    bool sample_valid = in_domain3 ( uvw_next, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));

    // if iso threshold reached -> search for accurate intersection
    if (  ( target_function ( last_sample_attribute.x ) - target_function ( iso_threshold ) ) * 
          ( target_function ( attrib.x )                - target_function ( iso_threshold ) ) < 0.0 &&
          ( sample_valid && last_sample_valid )
          )
    {
      binary_search_for_isosurface ( volumepointsbuffer, 
                                     attributepointsbuffer, 
                                     point_base_id,
                                     attribute_base_id,
                                     order, 
                                     ray_entry,
                                     ray_direction,
                                     uvw_current, 
                                     uvw_next, 
                                     last_sample_position, 
                                     sample_position, 
                                     last_sample_attribute, 
                                     attrib, 
                                     iso_threshold,
                                     newton_epsilon,
                                     max_iterations_newton,
                                     max_steps_binary_search,
                                     d1,
                                     d2,
                                     n1,
                                     n2,
                                     iso_hit_uvw, 
                                     iso_hit_position, 
                                     iso_hit_attribute,
                                     iso_hit_normal,
                                     total_samples );

      if ( validate_isosurface_intersection ( volumepointsbuffer, attributepointsbuffer, point_base_id, attribute_base_id, order, ray_entry, ray_direction, *iso_hit_uvw, iso_threshold, fixed_newton_epsilon) )
      {
        *next_uvw       = uvw_next;
        *next_ray_entry = sample_position;
        return true;
      }
    }

    // store last sample information
    last_sample_position  = sample_position;
    last_sample_attribute = attrib;
    last_sample_valid     = sample_valid;
    uvw_current           = uvw_next;

    // compute next sampling position
    if ( adaptive_sampling )
    {
      sample_position = compute_sampling_position_adaptively( sample_position,
                                                              ray_direction,
                                                              volume_bbox_diameter,
                                                              min_sample_distance,
                                                              max_sample_distance,
                                                              adaptive_sample_scale,
                                                              iso_threshold,
                                                              uvw_current,
                                                              attrib,
                                                              pdu, pdv, pdw,
                                                              adu, adv, adw );
    } else {
      sample_position = compute_sampling_position           ( sample_position,
                                                              ray_direction,
                                                              min_sample_distance,
                                                              volume_bbox_diameter );
    }
        
    if ( sample_end_reached )
    {
      return false;
    }

    if ( length(float3_t(sample_position.x, sample_position.y, sample_position.z) - float3_t(ray_entry.x, ray_entry.y, ray_entry.z) ) > sample_distance )
    {
      sample_end_reached = true;
      sample_position    = ray_exit;
      // do last iteration
    }
  } // for all samples in volume try to find iso surface

  return false;
}


__device__ inline
bool search_volume_for_iso_surface ( float4 const*    volumepointsbuffer,
                                     float2 const*    attributepointsbuffer,
                                     int              point_base_id,
                                     int              attribute_base_id,
                                     sample           last_sample,
                                     sample&          current_sample,
                                     sample&          iso_sample,
                                     ray_state const& ray,
                                     float4 const&    ray_exit,
                                     float            iso_threshold,
                                     bool             adaptive_sampling,
                                     float            min_sample_distance,
                                     float            max_sample_distance,
                                     float            adaptive_sample_scale,
                                     bool             adaptive_newton,
                                     float            fixed_newton_epsilon,
                                     unsigned         max_iterations_newton,
                                     unsigned         max_steps_binary_search,
                                     unsigned&        total_samples )
{
  // create ray in object coordinates
  float  sample_distance = length(current_sample.p - ray_exit);

  int    max_samples     = (int)( ceil ( sample_distance / ( min_sample_distance * current_sample.volume.volume_bbox_size ) ) );
  max_samples            = max ( 2, max_samples ); // use at least first entry and exit point
  max_samples            = min ( 256, max_samples );

  // initialize sampling by allocating variables and evaluating volume
  iso_sample             = last_sample;
  current_sample         = last_sample;
                         
  float4 ray_entry       = last_sample.p;
  float3 sample_uvw      = current_sample.uvw;
  float4 sample_position = current_sample.p;

  bool   sample_end_reached    = false;

  /********************************************************************************
  * search fragment interval for iso surface
  ********************************************************************************/  
  for ( int i = 0; i != max_samples; ++i, ++total_samples )
  {
    float newton_epsilon = 0.0f;

    if ( adaptive_newton )
    {
    } else {
      newton_epsilon  = fixed_newton_epsilon;
    }

    newton_volume(  volumepointsbuffer, 
                    point_base_id,
                    current_sample.volume.volume_order,
                    sample_uvw,
                    current_sample.uvw,
                    sample_position,
                    current_sample.p,
                    current_sample.dp_du,
                    current_sample.dp_dv,
                    current_sample.dp_dw,
                    ray.n1,
                    ray.n2,
                    ray.direction,
                    ray.d1,
                    ray.d2,
                    newton_epsilon,
                    max_iterations_newton );

    horner_volume_derivatives<float2, 1> ( attributepointsbuffer, 
                                           attribute_base_id, 
                                           current_sample.volume.volume_order, 
                                           current_sample.uvw, 
                                           current_sample.a, 
                                           current_sample.da_du, 
                                           current_sample.da_dv, 
                                           current_sample.da_dw );  

    // if iso threshold reached -> search for accurate intersection
    if (  ( current_sample.a.x - iso_threshold ) * ( last_sample.a.x - iso_threshold ) < 0.0 &&
          i > 0 &&
          ( current_sample.uvw_in_domain() && last_sample.uvw_in_domain() ) 
          )
    {
      
      binary_search_for_isosurface ( volumepointsbuffer,
                                     attributepointsbuffer,
                                     point_base_id,
                                     attribute_base_id,
                                     current_sample,
                                     last_sample,
                                     iso_sample,
                                     ray,
                                     iso_threshold,
                                     newton_epsilon,
                                     max_iterations_newton,
                                     max_steps_binary_search,
                                     total_samples );
     
      if ( iso_sample.uvw_in_domain() && 
           point_ray_distance(float3_t(ray.origin.x, ray.origin.y, ray.origin.z),
                              ray.direction,
                              float3_t(iso_sample.p.x, iso_sample.p.y, iso_sample.p.z) ) < fixed_newton_epsilon )
      {
        return true;
      }
    }

    // store last sample information
    last_sample = current_sample;

    // compute next sampling position
    if ( adaptive_sampling )
    {
      sample_uvw      = current_sample.uvw;
      sample_position = compute_sampling_position_adaptively( current_sample.p,
                                                               float4_t(ray.direction.x, ray.direction.y, ray.direction.z, 0.0f),
                                                               last_sample.volume.volume_bbox_size,
                                                               min_sample_distance,
                                                               max_sample_distance,
                                                               adaptive_sample_scale,
                                                               iso_threshold,
                                                               current_sample,
                                                               false, 
                                                               0xFFFF );
    } else {
      sample_uvw      = current_sample.uvw;
      sample_position = compute_sampling_position           ( last_sample.p, 
                                                              float4_t(ray.direction.x, ray.direction.y, ray.direction.z, 0.0f),
                                                              min_sample_distance,
                                                              last_sample.volume.volume_bbox_size );
    }
        
    if ( sample_end_reached )
    {
      return false;
    }

    if ( length(float3_t(sample_position.x, sample_position.y, sample_position.z) - float3_t(ray_entry.x, ray_entry.y, ray_entry.z) ) > sample_distance )
    {
      sample_end_reached = true;
      sample_position    = ray_exit;
      // do last iteration
    }
  } // for all samples in volume try to find iso surface

  return false;
}


#endif // LIBGPUCAST_SEARCH_VOLUME_FOR_ISO_SURFACE_H