/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : secant_search_for_isosurface.frag
*  project    : gpucast
*  description:
*
********************************************************************************/


/////////////////////////////////////////////////////////////////////////////
void secant_search_for_isosurface ( in samplerBuffer volumebuffer, 
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

  vec4 data_sample  = vec4(0.0);

  vec3 uvw          = vec3(0.0);
  vec4 point        = vec4(0.0);
  
  vec4 pmin         = first_sample_position;
  vec4 pmax         = last_sample_position;

  vec4 dmin         = first_sample_data;
  vec4 dmax         = last_sample_data;

  vec3 uvwmin       = first_uvw;
  vec3 uvwmax       = last_uvw;

  bool increasing_target_function = target_function ( first_sample_data ) < target_function ( last_sample_data );

  do
  {
    float alpha = abs(target_function(dmin) - target_function(iso_threshold)) /
                  abs(target_function(dmin) - target_function(dmax));

    alpha = clamp (alpha, 0.05, 0.95);

    vec3 uvwstart  = mix(uvwmin, uvwmax, alpha);
    point          = mix(pmin, pmax, alpha); 

    vec4 tmp = vec4(0.0);
    vec4 du  = vec4(0.0);
    vec4 dv  = vec4(0.0);
    vec4 dw  = vec4(0.0);

    newton_volume  ( volumebuffer, volume_index, order.x, order.y, order.z, uvwstart, uvw, point, tmp, du, dv, dw, n1, n2, ray_direction.xyz, d1, d2, newton_epsilon, newton_max_iterations );
    evaluateVolume ( databuffer,   data_index,   order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, data_sample );

    if ( target_function ( data_sample ) < target_function ( iso_threshold ) == increasing_target_function ) 
    {
      dmin = data_sample;
      pmin = point;
      uvwmin = uvw;
    } else {
      dmax = data_sample;
      pmax = point;
      uvwmax = uvw;
    }

    ++iterations;

  } while ( length ( uvwmin - uvwmax ) < 0.01 &&
            iterations < max_steps_binary_search );

  iso_uvw       = uvw;
  iso_data      = data_sample;
  iso_position  = point;
  evaluateVolume ( volumebuffer, volume_index, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, iso_position );
}