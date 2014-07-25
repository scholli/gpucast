#ifndef LIBGPUCAST_CUDA_CROSS_H
#define LIBGPUCAST_CUDA_CROSS_H

__device__
inline float3 cross ( float3 const& lhs, float3 const& rhs ) 
{
  return float3_t ( lhs.y * rhs.z - lhs.z * rhs.y,
                    lhs.z * rhs.x - lhs.x * rhs.z,
                    lhs.x * rhs.y - lhs.y * rhs.x );
}

__device__
inline float4 cross ( float4 const& lhs, float4 const& rhs ) 
{
  return float4_t ( lhs.y * rhs.z - lhs.z * rhs.y,
                    lhs.z * rhs.x - lhs.x * rhs.z,
                    lhs.x * rhs.y - lhs.y * rhs.x, 0.0f );
}



#endif // LIBGPUCAST_CUDA_CROSS_H