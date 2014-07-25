#ifndef LIB_GPUCAST_PROJECT_TO_EUCLIDIAN_H
#define LIB_GPUCAST_PROJECT_TO_EUCLIDIAN_H

/*********************************************************************
 * projects point back to euclidian space : p'homogenous[x/w, y/w, z/w, w] => p'euclid[x,y,z,w]
 *********************************************************************/
__device__ inline
float3 float3_to_euclid ( float3 const& point )
{
  float weight  = point.z;
  point        /= weight;
  point.z       = weight;
  return point;
}

__device__ inline
float4 float4_to_euclid ( float4 const& point )
{
  float weight = point.w;
  point       /= weight;
  point.w      = weight;
  return point;
}

#endif // LIB_GPUCAST_PROJECT_TO_EUCLIDIAN_H

