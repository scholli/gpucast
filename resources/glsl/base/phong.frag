/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : phong.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// phong shading:
//   point, normal and lightpos in camera coordinates
///////////////////////////////////////////////////////////////////////////////
vec4 phong_shading ( in vec4 point,
                     in vec4 normal,
                     in vec4 lightpos )
{
  vec3 L = normalize ( lightpos.xyz - point.xyz );
  vec3 N = normalize ( normal.xyz );

  N = faceforward( -point.xyz, N ); // faceforward doesn't work properly

  float diffuse = max( 0.0, dot (N , L));
  vec4  color   = vec4(diffuse, diffuse, diffuse, 1.0);
  return color;
}