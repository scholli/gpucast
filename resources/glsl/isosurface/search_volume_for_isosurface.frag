/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : search_volume_for_iso_surface.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_SEARCH_VOLUME_FOR_ISO_SURFACE_FRAG
#define LIBGPUCAST_SEARCH_VOLUME_FOR_ISO_SURFACE_FRAG

///////////////////////////////////////////////////////////////////////////////
bool search_volume_for_iso_surface ( in samplerBuffer pointbuffer,
                                     in samplerBuffer attributebuffer,
                                     in int           point_base_id,
                                     in int           attribute_base_id,
                                     in uvec3         order,
                                     in vec3          start_uvw,
                                     in vec4          iso_threshold,
                                     in vec4          ray_entry,
                                     in vec4          ray_exit,
                                     in bool          adaptive_sampling,
                                     in float         volume_bbox_diameter,
                                     in float         min_sample_distance,
                                     in float         max_sample_distance,
                                     in float         adaptive_sample_scale,
                                     in bool          adaptive_newton,
                                     in float         fixed_newton_epsilon,
                                     in uint          max_iterations_newton,
                                     in uint          max_steps_binary_search,
                                     out vec4         iso_hit_position,
                                     out vec4         iso_hit_attribute,
                                     out vec4         iso_hit_normal,
                                     out vec3         iso_hit_uvw,
                                     out vec4         next_ray_entry,
                                     out vec3         next_uvw )
{
  iso_hit_position  = vec4(0.0f);
  iso_hit_attribute = vec4(0.0f); 
  iso_hit_normal    = vec4(0.0f);
  iso_hit_uvw       = vec3(0.0f);

  // create ray in object coordinates
  vec4 ray_direction     = vec4(ray_exit.xyz - ray_entry.xyz, 0.0f);
  float  sample_distance = length(ray_direction);
  int    max_samples     = (int)( ceil ( sample_distance / ( min_sample_distance * volume_bbox_diameter ) ) );

  // normalize ray direction
  ray_direction         /= sample_distance;  

  // convert ray to intersection of two planes
  vec3  n1   = vec3(0.0f);
  vec3  n2   = vec3(0.0f);
  float d1   = 0.0f;
  float d2   = 0.0f;

  raygen ( ray_entry, ray_direction, n1, n2, d1, d2 );

  // initialize sampling by allocating variables and evaluating volume
  vec3 uvw = start_uvw;

  vec4 point,  pdu, pdv, pdw;
  vec4 attrib, adu, adv, adw;
  vec3 uvw_current           = uvw; 
  vec3 uvw_next              = vec3(0.0f);
  vec4 sample_position       = ray_entry;
  bool sample_end_reached    = false;
  bool last_sample_valid     = false;
  vec4 last_sample_attribute = vec4(0.0f);
  vec4 last_sample_position  = vec4(0.0f);

  /********************************************************************************
  * search fragment interval for iso surface
  ********************************************************************************/  
  for ( int i = 0; i != max_samples; ++i )
  {
    float newton_epsilon = 0.0f;

    if ( adaptive_newton ) 
    {

    } else {
     newton_epsilon      = fixed_newton_epsilon;
    }

    bool newton_success = newton_volume ( pointbuffer, 
                                          point_base_id,
                                          int(order.x),
                                          int(order.y),
                                          int(order.z),
                                          uvw_current,
                                          uvw_next,
                                          sample_position,
                                          point,
                                          pdu,
                                          pdv,
                                          pdw,
                                          n1,
                                          n2,
                                          ray_direction.xyz,
                                          d1,
                                          d2,
                                          newton_epsilon,
                                          int(max_iterations_newton) );

    evaluateVolume ( pointbuffer,     point_base_id,     int(order.x), int(order.y), int(order.z), uvw_next.x, uvw_next.y, uvw_next.z, point,  pdu, pdv, pdw );  
    evaluateVolume ( attributebuffer, attribute_base_id, int(order.x), int(order.y), int(order.z), uvw_next.x, uvw_next.y, uvw_next.z, attrib, adu, adv, adw );  

    bool sample_valid = in_bezier_domain ( uvw_next );

    // if iso threshold reached -> search for accurate intersection
    if (  ( target_function ( last_sample_attribute) - target_function ( iso_threshold ) ) * 
          ( target_function ( attrib )               - target_function ( iso_threshold ) ) < 0.0 &&
          ( sample_valid && last_sample_valid )
          )
    {
      binary_search_for_isosurface ( pointbuffer, 
                                     attributebuffer, 
                                     point_base_id,
                                     attribute_base_id,
                                     ivec3(order), 
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
                                     int(max_iterations_newton),
                                     int(max_steps_binary_search),
                                     d1,
                                     d2,
                                     n1,
                                     n2,
                                     iso_hit_uvw, 
                                     iso_hit_position, 
                                     iso_hit_attribute,
                                     iso_hit_normal );

      if ( validate_isosurface_intersection ( pointbuffer, attributebuffer, ivec4(point_base_id, order), ray_entry, ray_direction, iso_hit_uvw, iso_threshold, 0.0005f) )
      {
        next_uvw       = uvw_next;
        next_ray_entry = sample_position;
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

    if ( length(sample_position.xyz - ray_entry.xyz) > sample_distance )
    {
      sample_end_reached = true;
    }
  } // for all samples in volume try to find iso surface

  return false;
}

#endif