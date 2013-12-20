/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_iso_normal.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

/////////////////////////////////////////////////////////////////////////////
vec4 compute_iso_normal ( in vec4   du, 
                          in vec4   dv, 
                          in vec4   dw, 
                          in vec4   ddu, 
                          in vec4   ddv, 
                          in vec4   ddw)
{
  // partial derivative dD / duvw
  vec3 dD_duvw = normalize( vec3 (target_function(ddu), 
                                  target_function(ddv),
                                  target_function(ddw) ) );

  // solutions for 0 = dD / duvw * duvw0
  vec3 duvw0 = vec3 ( 1.0, -dD_duvw.x/dD_duvw.y, 0.0);
  vec3 duvw1 = vec3 ( 0.0, -dD_duvw.z/dD_duvw.y, 1.0);

  mat3 M      = mat3(du.xyz, dv.xyz, dw.xyz);

  vec3 dxyz0  = M * duvw0;
  vec3 dxyz1  = M * duvw1;

  vec3 isonormal  = normalize ( cross ( dxyz0, dxyz1 ) );

  return vec4(isonormal, 0.0);
}