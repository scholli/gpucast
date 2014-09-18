/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : binary_search_for_iso_surface.frag
*  project    : gpucast
*  description:
*
********************************************************************************/


/////////////////////////////////////////////////////////////////////////////
void binary_search_for_isosurface (  in samplerBuffer volumebuffer, 
                                     in samplerBuffer databuffer,
                                     in int           volume_index,
                                     in int           data_index,
                                     in ivec3         order, 
                                     in vec4          ray_entry,
                                     in vec4          ray_direction,
                                     in vec3          first_uvw, 
                                     in vec3          last_uvw, 
                                     in vec4          first_sample_position, 
                                     in vec4          last_sample_position, 
                                     in vec4          first_sample_data, 
                                     in vec4          last_sample_data, 
                                     in vec4          search_iso_value,
                                     in float         newton_epsilon,
                                     in int           newton_max_iterations,
                                     in int           max_steps_binary_search,
                                     in float         d1,
                                     in float         d2,
                                     in vec3          n1,
                                     in vec3          n2,
                                     out vec3         iso_uvw,
                                     out vec4         iso_position, 
                                     out vec4         iso_data,
                                     out vec4         iso_normal )
{
  
#if 0
  // approximate binary search by using a simle linear interpolation
  float interp  = (target_function(search_iso_value) - target_function(first_sample_data)) / (target_function(last_sample_data) - target_function(first_sample_data));

  iso_data      = mix(last_sample_data,     first_sample_data,     interp);
  iso_uvw       = mix(last_uvw,             first_uvw,             interp);
  iso_position  = mix(last_sample_position, first_sample_position, interp);
#else

  // binary search the corresponding uvw-interval -> ignore geometric position
  int  iterations   = 0;

  vec4 attrib, adu, adv, adw;
  vec4 p, du, dv, dw;

  vec3 uvwstart     = mix(first_uvw, last_uvw, 0.5);
  vec3 uvw          = vec3(0.0);
  vec4 point        = vec4(0.0);
  
  vec4 pmin         = first_sample_position;
  vec4 pmax         = last_sample_position;

  bool increasing_target_function = target_function ( first_sample_data ) < target_function ( last_sample_data );

  do
  {
    point = mix(pmin, pmax, 0.5); 

    newton_volume  ( volumebuffer, volume_index, order.x, order.y, order.z, uvwstart, uvw, point, p, du, dv, dw, n1, n2, ray_direction.xyz, d1, d2, newton_epsilon, newton_max_iterations );
    evaluateVolume ( databuffer,   data_index,   order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, attrib, adu, adv, adw );

    if ( target_function ( attrib ) < target_function ( search_iso_value) == increasing_target_function ) 
    {
      pmin = point;
    } else {
      pmax = point;
    }

    uvwstart = uvw;

    ++iterations;

  } while ( iterations < max_steps_binary_search );

  iso_uvw      = uvw;
  iso_data     = attrib;
  iso_normal   = compute_iso_normal(du, dv, dw, adu, adv, adw);
  iso_position = point;

#endif
}


