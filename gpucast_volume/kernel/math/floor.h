#ifndef LIBGPUCAST_CUDA_FLOOR_H
#define LIBGPUCAST_CUDA_FLOOR_H

__device__ inline
float4 floor ( float4 const& v ) 
{
  return float4_t ( floor (v.x), floor (v.y), floor (v.z), floor (v.w) );
}

__device__ inline
float3 floor ( float3 const& v ) 
{
  return float3_t ( floor (v.x), floor (v.y), floor (v.z) );
}

__device__ inline
float2 floor ( float2 const& v ) 
{
  return float2_t ( floor (v.x), floor (v.y) );
}

#endif // LIBGPUCAST_CUDA_FLOOR_H