#ifndef LIBGPUCAST_CUDA_MAX_H
#define LIBGPUCAST_CUDA_MAX_H

__device__ inline 
float4 max ( float4 const& lhs, float4 const& rhs ) 
{
  return float4_t ( max (lhs.x, rhs.x), 
                    max (lhs.y, rhs.y),
                    max (lhs.z, rhs.z),
                    max (lhs.w, rhs.w) );
}

__device__ inline 
float3 max ( float3 const& lhs, float3 const& rhs ) 
{
  return float3_t ( max (lhs.x, rhs.x), 
                    max (lhs.y, rhs.y),
                    max (lhs.z, rhs.z) );
}

__device__ inline 
float2 max ( float2 const& lhs, float2 const& rhs ) 
{
  return float2_t ( max (lhs.x, rhs.x), 
                    max (lhs.y, rhs.y) );
}

#endif // LIBGPUCAST_CUDA_MAX_H