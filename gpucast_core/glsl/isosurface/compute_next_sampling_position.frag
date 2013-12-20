/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_next_sampling_position.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

/////////////////////////////////////////////////////////////////////////////
vec4 compute_sampling_position_adaptively ( in vec4   last_sample_position,
                                            in vec4   ray_direction,
                                            in float  obb_diameter,
                                            in float  min_sample_distance,
                                            in float  max_sample_distance,
                                            in float  adaptive_sample_scale,
                                            in vec4   iso_value,
                                            in vec4   last_sample_data,
                                            in vec4   du, 
                                            in vec4   dv, 
                                            in vec4   dw,
                                            in vec4   ddu, 
                                            in vec4   ddv, 
                                            in vec4   ddw )
{
  // the partial derivatives dV/du, dV/dv and dV/dw constitute thecoordinate system V
  mat3 V        = mat3 ( du.xyz, dv.xyz, dw.xyz );
  mat3 Vinv     = inverse   ( V );
  mat3 VinvT    = transpose ( Vinv );



  // partial data derivatives: dD/du, dD/dv, dD/dw transformed into one-dimensional iso space
  vec3 dD_duvw  = vec3 ( target_function(ddu), 
                         target_function(ddv),
                         target_function(ddw) );

  // transform data derivative into object space -> how much does D change if you go along the ray (dx,dy,dz)
  vec3 dD_dxyz  = VinvT * dD_duvw;

  // dot product with ray indicates -> how does D change if you step 1 along the ray
  float dD_dt   = dot ( dD_dxyz, ray_direction.xyz);

  // compute distance to isosurface in data space
  float dD0     = target_function(iso_value) - target_function(last_sample_data);

  float adaptive_stepwidth = 0.0;
  
  // compute step
  if ( dD_dt * dD0 <= 0.0 ) 
  {
    // ray and isosurface diverge -> go max step according to bbox diameter
    adaptive_stepwidth =  obb_diameter * max_sample_distance;
  } else {
    // ray and isosurface seem to converge -> try to solve equation assuming linearity
    adaptive_stepwidth =  adaptive_sample_scale * dD0 / dD_dt ;
  }
  
  // clamp stepwidth
  adaptive_stepwidth      = clamp ( adaptive_stepwidth, obb_diameter * min_sample_distance, obb_diameter * max_sample_distance );
 
  // compute next sample position
  return last_sample_position + ray_direction * adaptive_stepwidth;
}


/////////////////////////////////////////////////////////////////////////////
vec4 compute_sampling_position (  in vec4   last_sample_position,
                                  in vec4   ray_direction,
                                  in float  min_sample_distance,
                                  in float  obb_diameter )
{
  vec4 fixed_stepwidth = ray_direction * obb_diameter * min_sample_distance;
  return last_sample_position + fixed_stepwidth;
}
