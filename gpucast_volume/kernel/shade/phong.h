#ifndef LIB_GPUCAST_PHONG_H
#define LIB_GPUCAST_PHONG_H

#include "shade/faceforward.h"

__device__ inline
float4 phong_shading ( float4 const& point,
                       float4 const& normal,
                       float4 const& lightpos )
{
  float4 to_light = lightpos - point;
  float3 L        = normalize ( float3_t(to_light.x, to_light.y, to_light.z) );
  float3 N        = normalize ( float3_t(normal.x, normal.y, normal.z) );

  N = faceforward( float3_t(-point.x, -point.y, -point.z), N ); // faceforward doesn't work properly

  float diffuse = fmax( 0.0f, dot (N , L));
  float4  color = float4_t(diffuse, diffuse, diffuse, 1.0);
  return color;
}

#endif // LIB_GPUCAST_PHONG_H
