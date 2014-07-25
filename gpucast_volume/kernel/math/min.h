#ifndef LIBGPUCAST_CUDA_MIN_H
#define LIBGPUCAST_CUDA_MIN_H

__device__ inline 
float4 min ( float4 const& lhs, float4 const& rhs ) 
{
  return float4_t ( min (lhs.x, rhs.x), 
                    min (lhs.y, rhs.y),
                    min (lhs.z, rhs.z),
                    min (lhs.w, rhs.w) );
}

__device__ inline 
float3 min ( float3 const& lhs, float3 const& rhs ) 
{
  return float3_t ( min (lhs.x, rhs.x), 
                    min (lhs.y, rhs.y),
                    min (lhs.z, rhs.z) );
}

__device__ inline 
float2 min ( float2 const& lhs, float2 const& rhs ) 
{
  return float2_t ( min (lhs.x, rhs.x), 
                    min (lhs.y, rhs.y) );
}

#endif // LIBGPUCAST_CUDA_MIN_H