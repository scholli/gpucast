/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : validate_isosurface_intersection.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

#include "./libgpucast/glsl/isosurface/in_bezier_domain.frag"
#include "./libgpucast/glsl/isosurface/point_ray_distance.frag"

/////////////////////////////////////////////////////////////////////////////
bool validate_isosurface_intersection ( in samplerBuffer volumebuffer,
                                        in samplerBuffer attributebuffer,
                                        in ivec4         volume,
                                        in vec4          ray_entry,
                                        in vec4          ray_direction,
                                        in vec3          uvw,
                                        in vec4          isovalue,
                                        in float         epsilon )
{
  vec4 point, du, dv, dw;
  vec4 data, ddu, ddv, ddw;

  evaluateVolume ( volumebuffer,    volume.x, volume.y, volume.z, volume.w, uvw.x, uvw.y, uvw.z, point );
  //evaluateVolume ( attributebuffer, volume.x, volume.y, volume.z, volume.w, uvw.x, uvw.y, uvw.z, data, ddu, ddv, ddw );
  //float relative_error = fabs( target_function(data) - target_function(iso_threshold) ) / fabs( target_function( data ));

  return //relative_error < intersection_epsilon &&
         in_bezier_domain(uvw) && 
         point_ray_distance(ray_entry.xyz, ray_direction.xyz, point.xyz) < epsilon;
}
