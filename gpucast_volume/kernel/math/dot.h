#ifndef LIBGPUCAST_CUDA_DOT_H
#define LIBGPUCAST_CUDA_DOT_H

__device__
inline float dot ( float2 const lhs, float2 const& rhs ) 
{
  return lhs.x * rhs.x + lhs.y * rhs.y;
}

__device__
inline float dot ( float3 const lhs, float3 const& rhs ) 
{
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__
inline float dot ( float4 const lhs, float4 const& rhs ) 
{
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}

#endif // LIBGPUCAST_CUDA_DOT_H