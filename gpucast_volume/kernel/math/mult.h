#ifndef LIB_GPUCAST_MULT_H
#define LIB_GPUCAST_MULT_H

#include "math/matrix_layout.h"

#ifdef COL_MAJOR

/*********************************************************************
 * multiplication of a 2x2 matrix
 *********************************************************************/
__device__ 
inline void 
mult_mat2 ( float2* a, float2* b, float2* c) 
{
  c[0].x = a[0].x * b[0].x + a[1].x * b[0].y;
  c[0].y = a[0].y * b[0].x + a[1].y * b[0].y;

  c[1].x = a[0].x * b[1].x + a[1].x * b[1].y;
  c[1].y = a[0].y * b[1].x + a[1].y * b[1].y;
}

__device__
inline float2
mult_mat2_float2 ( float2* m, float2 p )
{
  return float2_t ( m[0].x * p.x + m[1].x * p.y,
                       m[0].y * p.x + m[1].y * p.y );
}

/*********************************************************************
 * multiplication of a 3x3 matrix
 *********************************************************************/
__device__ 
inline void 
mult_mat3 ( float3 const* a, float3 const* b, float3* c) 
{
  c[0].x = a[0].x * b[0].x + a[1].x * b[0].y + a[2].x * b[0].z;
  c[0].y = a[0].y * b[0].x + a[1].y * b[0].y + a[2].y * b[0].z;
  c[0].z = a[0].z * b[0].x + a[1].z * b[0].y + a[2].z * b[0].z;

  c[1].x = a[0].x * b[1].x + a[1].x * b[1].y + a[2].x * b[1].z;
  c[1].y = a[0].y * b[1].x + a[1].y * b[1].y + a[2].y * b[1].z;
  c[1].z = a[0].z * b[1].x + a[1].z * b[1].y + a[2].z * b[1].z;

  c[2].x = a[0].x * b[2].x + a[1].x * b[2].y + a[2].x * b[2].z;
  c[2].y = a[0].y * b[2].x + a[1].y * b[2].y + a[2].y * b[2].z;
  c[2].z = a[0].z * b[2].x + a[1].z * b[2].y + a[2].z * b[2].z;
}

__device__ 
inline float3
mult_mat3_float3 ( float3 const* m, float3 const& p )
{
  return float3_t ( m[0].x * p.x + m[1].x * p.y + m[2].x * p.z,
                    m[0].y * p.x + m[1].y * p.y + m[2].y * p.z,
                    m[0].z * p.x + m[1].z * p.y + m[2].z * p.z );
}

__device__ inline float3 
operator* ( float3 const* m, float3 const& p )
{
  return float3_t ( m[0].x * p.x + m[1].x * p.y + m[2].x * p.z,
                    m[0].y * p.x + m[1].y * p.y + m[2].y * p.z,
                    m[0].z * p.x + m[1].z * p.y + m[2].z * p.z );
}

/*********************************************************************
 * multiplication of a 4x4 matrix
 *********************************************************************/

__device__ 
inline float4 
mult_mat4_float4 ( float4 const* m, float4 const& p )
{
  return float4_t ( m[0].x * p.x + m[1].x * p.y + m[2].x * p.z + m[3].x * p.w,
                    m[0].y * p.x + m[1].y * p.y + m[2].y * p.z + m[3].y * p.w,
                    m[0].z * p.x + m[1].z * p.y + m[2].z * p.z + m[3].z * p.w,
                    m[0].w * p.x + m[1].w * p.y + m[2].w * p.z + m[3].w * p.w);
}

__device__ inline float4 
operator* ( float4 const* m, float4 const& p )
{
  return float4_t ( m[0].x * p.x + m[1].x * p.y + m[2].x * p.z + m[3].x * p.w,
                    m[0].y * p.x + m[1].y * p.y + m[2].y * p.z + m[3].y * p.w,
                    m[0].z * p.x + m[1].z * p.y + m[2].z * p.z + m[3].z * p.w,
                    m[0].w * p.x + m[1].w * p.y + m[2].w * p.z + m[3].w * p.w);
}

#endif


#ifdef ROW_MAJOR

/*********************************************************************
 * multiplication of a 2x2 matrix
 *********************************************************************/
__device__
void
mult_mat2 ( float2* a, float2* b, float2* c) 
{
  c[0].x = dot(a[0], (float2)(b[0].x, b[1].x));
  c[0].y = dot(a[0], (float2)(b[0].y, b[1].y));

  c[1].x = dot(a[1], (float2)(b[0].x, b[1].x));
  c[1].y = dot(a[1], (float2)(b[0].y, b[1].y));
}

float2
mult_mat2_float2 ( float2* m, float2 p )
{
  return (float2)( dot(m[0], p), dot(m[1], p));
}

/*********************************************************************
 * multiplication of a 3x3 matrix
 *********************************************************************/
__device__
void
mult_mat3 ( float3* a, float3* b, float3* c) 
{
  c[0].x = dot(a[0], (float3)(b[0].x, b[1].x, b[2].x));
  c[0].y = dot(a[0], (float3)(b[0].y, b[1].y, b[2].y));
  c[0].z = dot(a[0], (float3)(b[0].z, b[1].z, b[2].z));

  c[1].x = dot(a[1], (float3)(b[0].x, b[1].x, b[2].x));
  c[1].y = dot(a[1], (float3)(b[0].y, b[1].y, b[2].y));
  c[1].z = dot(a[1], (float3)(b[0].z, b[1].z, b[2].z));

  c[2].x = dot(a[2], (float3)(b[0].x, b[1].x, b[2].x));
  c[2].y = dot(a[2], (float3)(b[0].y, b[1].y, b[2].y));
  c[2].z = dot(a[2], (float3)(b[0].z, b[1].z, b[2].z));
}

__device__
float3
mult_mat3_float3 ( float3* m, float3 p )
{
  return (float3)( dot(m[0], p), dot(m[1], p), dot(m[2], p) );
}

/*********************************************************************
 * multiplication of a 4x4 matrix
 *********************************************************************/
__device__
float4
mult_mat4_float4 ( float4* m, float4 p )
{
  return (float4)( dot(m[0], p), dot(m[1], p), dot(m[2], p), dot(m[3], p) );
}

#endif // ROW_MAJOR

#endif // LIB_GPUCAST_MULT_H