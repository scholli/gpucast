/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trilinear_interpolation.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

///////////////////////////////////////////////////////////////////////////////
vec3 trilinear_interpolation ( in samplerBuffer point_data,
                               in int           start_index,
                               in vec3          uvw )
{
  vec4 p000 = texelFetchBuffer(point_data, start_index    );
  vec4 p100 = texelFetchBuffer(point_data, start_index + 1);
  vec4 p010 = texelFetchBuffer(point_data, start_index + 2);
  vec4 p110 = texelFetchBuffer(point_data, start_index + 3);

  vec4 p001 = texelFetchBuffer(point_data, start_index + 4);
  vec4 p101 = texelFetchBuffer(point_data, start_index + 5);
  vec4 p011 = texelFetchBuffer(point_data, start_index + 6);
  vec4 p111 = texelFetchBuffer(point_data, start_index + 7);

  vec4 pu0  = mix(p000, p100, uvw.x);
  vec4 pu1  = mix(p010, p110, uvw.x);
  vec4 pu2  = mix(p001, p101, uvw.x);
  vec4 pu3  = mix(p011, p111, uvw.x);

  vec4 puv0 = mix(pu0, pu1, uvw.y);
  vec4 puv1 = mix(pu2, pu3, uvw.y);

  return mix(puv0, puv1, uvw.z).xyz;
}

