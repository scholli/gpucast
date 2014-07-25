#ifndef LIB_GPUCAST_CUDA_OPERATOR_H
#define LIB_GPUCAST_CUDA_OPERATOR_H

// float4 operator
__device__ inline
float4 operator+ (float4 const& lhs, float4 const& rhs)
{
  return float4_t(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
__device__ inline
float4 operator- (float4 const& lhs, float4 const& rhs)
{
  return float4_t(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}
__device__ inline
float4 operator- (float4 const& lhs)
{
  return float4_t(-lhs.x, -lhs.y, -lhs.z, -lhs.w);
}
__device__ inline
float4 operator* (float lhs, float4 const& rhs)
{
  return float4_t(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}
__device__ inline
float4 operator* (float4 const& lhs, float rhs)
{
  return float4_t(rhs * lhs.x, rhs * lhs.y, rhs * lhs.z, rhs * lhs.w);
}
__device__ inline
float4 operator* (float4 const& lhs, float4 rhs)
{
  return float4_t(rhs.x * lhs.x, rhs.y * lhs.y, rhs.z * lhs.z, rhs.w * lhs.w);
}
__device__ inline
float4 operator/ (float4 const& lhs, float rhs)
{
  return float4_t( lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
__device__ inline
float4 operator/ (float4 const& lhs, float4 const& rhs)
{
  return float4_t( lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

// float3 operator
__device__ inline
float3 operator+ (float3 const& lhs, float3 const& rhs)
{
  return float3_t(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
__device__ inline
float3 operator- (float3 const& lhs, float3 const& rhs)
{
  return float3_t(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
__device__ inline
float3 operator- (float3 const& lhs)
{
  return float3_t(-lhs.x, -lhs.y, -lhs.z);
}
__device__ inline
float3 operator* (float lhs, float3 const& rhs)
{
  return float3_t(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
__device__ inline
float3 operator* (float3 const& lhs, float rhs)
{
  return float3_t(rhs * lhs.x, rhs * lhs.y, rhs * lhs.z);
}
__device__ inline
float3 operator* (float3 const& lhs, float3 const& rhs)
{
  return float3_t(rhs.x * lhs.x, rhs.y * lhs.y, rhs.z * lhs.z);
}
__device__ inline
float3 operator/ (float3 const& lhs, float rhs)
{
  return float3_t(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
__device__ inline
float3 operator/ (float3 const& lhs, float3 const& rhs)
{
  return float3_t(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}


// float2 operator
__device__ inline
float2 operator+ (float2 const& lhs, float2 const& rhs)
{
  return float2_t(lhs.x + rhs.x, lhs.y + rhs.y);
}
__device__ inline
float2 operator- (float2 const& lhs, float2 const& rhs)
{
  return float2_t(lhs.x - rhs.x, lhs.y - rhs.y);
}
__device__ inline
float2 operator* (float lhs, float2 const& rhs)
{
  return float2_t(lhs * rhs.x, lhs * rhs.y);
}
__device__ inline
float2 operator* (float2 const& lhs, float rhs)
{
  return float2_t(rhs * lhs.x, rhs * lhs.y);
}
__device__ inline
float2 operator* (float2 const& lhs, float2 const& rhs)
{
  return float2_t(rhs.x * lhs.x, rhs.y * lhs.y);
}
__device__ inline
float2 operator/ (float2 const& lhs, float rhs)
{
  return float2_t(lhs.x / rhs, lhs.y / rhs);
}
__device__ inline
float2 operator/ (float2 const& lhs, float2 const& rhs)
{
  return float2_t(lhs.x / rhs.x, lhs.y / rhs.y);
}

// int2 operator
__device__ inline
int2 operator+ (int2 const& lhs, int2 const& rhs)
{
  return int2_t(lhs.x + rhs.x, lhs.y + rhs.y);
}

__device__ inline
int2 operator- (int2 const& lhs, int2 const& rhs)
{
  return int2_t(lhs.x - rhs.x, lhs.y - rhs.y);
}

__device__ inline
int2 operator* (float lhs, int2 const& rhs)
{
  return int2_t(lhs * rhs.x, lhs * rhs.y);
}

__device__ inline
int2 operator* (int2 const& lhs, float rhs)
{
  return int2_t(rhs * lhs.x, rhs * lhs.y);
}

__device__ inline
int2 operator* (int2 const& lhs, int2 const& rhs)
{
  return int2_t(rhs.x * lhs.x, rhs.y * lhs.y);
}

__device__ inline
int2 operator/ (int2 const& lhs, float rhs)
{
  return int2_t(lhs.x / rhs, lhs.y / rhs);
}

__device__ inline
int2 operator/ (int2 const& lhs, int2 const& rhs)
{
  return int2_t(lhs.x / rhs.x, lhs.y / rhs.y);
}



#endif // LIB_GPUCAST_CUDA_OPERATOR_H