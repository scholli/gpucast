/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : newton_search_for_isosurface.frag
*  project    : gpucast
*  description:
*
********************************************************************************/


/////////////////////////////////////////////////////////////////////////////
void newton_search_for_isosurface ( in samplerBuffer volumebuffer, 
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
                                    in vec4          iso_threshold,
                                    in float         newton_epsilon,
                                    in int           newton_max_iterations,
                                    in int           max_steps_binary_search,
                                    in float         d1,
                                    in float         d2,
                                    in vec3          n1,
                                    in vec3          n2,
                                    out vec3         iso_uvw, 
                                    out vec4         iso_position, 
                                    out vec4         iso_data )
{
  int  iterations   = 0;

  float alpha           = abs(target_function(first_sample_data) - target_function(iso_threshold)) /
                          abs(target_function(first_sample_data) - target_function(last_sample_data));
                        
  float sample_distance = length(first_sample_position.xyz - last_sample_position.xyz);

  vec3 uvwstart     = mix(first_uvw, last_uvw, alpha );
  vec3 uvw          = vec3(0.0);

  vec4 point        = mix(first_sample_position, last_sample_position, alpha );
  vec4 pdu          = vec4(0.0);
  vec4 pdv          = vec4(0.0);
  vec4 pdw          = vec4(0.0);
  vec4 tmp          = vec4(0.0);

  vec4 attrib       = vec4(0.0);
  vec4 adu          = vec4(0.0);
  vec4 adv          = vec4(0.0);
  vec4 adw          = vec4(0.0);
  
  float error       = 1.0;

  do
  {
    // determine domain parameter for sample point
    newton_volume  ( volumebuffer, volume_index, order.x, order.y, order.z, uvwstart, uvw, point, tmp, pdu, pdv, pdw, n1, n2, ray_direction.xyz, d1, d2, newton_epsilon, newton_max_iterations );
    uvwstart = uvw;

    // determine attribute value at sample point
    evaluateVolume ( databuffer,   data_index,   order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, attrib, adu, adv, adw );
    
    // quasi newton step to next sample point
    vec4 next_point = compute_sampling_position_adaptively ( point, ray_direction, sample_distance, 0.0, 1.0, 1.0, iso_threshold, attrib, pdu, pdv, pdw, adu, adv, adw );
    next_point = clamp ( next_point, min(first_sample_position, last_sample_position), max(first_sample_position, last_sample_position));
    uvwstart   = clamp ( uvwstart, vec3(0.0), vec3(1.0));
    
    error = length(next_point.xyz - point.xyz) / sample_distance;
    point = next_point;

    ++iterations;

  } while ( abs(error) < 0.01 &&
            iterations < max_steps_binary_search );

  iso_uvw       = uvw;
  iso_data      = attrib;
  iso_position  = point;
  evaluateVolume ( volumebuffer, volume_index, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, iso_position );
  evaluateVolume ( databuffer, data_index, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, iso_data );
}