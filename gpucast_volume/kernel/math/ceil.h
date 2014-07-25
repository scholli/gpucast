#ifndef LIBGPUCAST_CUDA_CEIL_H
#define LIBGPUCAST_CUDA_CEIL_H

__device__
inline float4 ceil ( float4 const& v ) 
{
  return float4_t ( ceil (v.x), ceil (v.y), ceil (v.z), ceil (v.w) );
}

__device__
inline float3 ceil ( float3 const& v ) 
{
  return float3_t ( ceil (v.x), ceil (v.y), ceil (v.z) );
}

__device__
inline float2 ceil ( float2 const& v ) 
{
  return float2_t ( ceil (v.x), ceil (v.y) );
}

#endif // LIBGPUCAST_CUDA_CEIL_H