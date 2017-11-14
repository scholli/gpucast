#ifndef LIBGPUCAST_CONVERSION_H
#define LIBGPUCAST_CONVERSION_H

#include <cuda_fp16.h>

///////////////////////////////////////////////////////////////////////////////
__device__ inline
int floatBitsToInt( float floatVal )
{
  int intVal = __float_as_int( floatVal );
  return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float intBitsToFloat( int intVal )
{
  return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline 
void unpack2short ( unsigned int input, unsigned short* x, unsigned short* y )
{
  *y = (input & 0xFFFF0000) >> 16U;
  *x = (input & 0x0000FFFF);
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void pack2short ( unsigned short x, unsigned short y, unsigned int* result )
{
  *result  = 0U;
  *result |= y << 16U;
  *result |= x;
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void pack_uvwx ( float* uvw, unsigned int* uv, unsigned int* wx )
{
  unsigned short u = __float2half_rn(uvw[0]);
  unsigned short v = __float2half_rn(uvw[1]);
  unsigned short w = __float2half_rn(uvw[2]);
    
  pack2short(u, v, uv);
  pack2short(w, 0, wx);
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
unsigned int pack_uv ( float2 uv )
{
  unsigned short u = __float2half_rn(uv.x);
  unsigned short v = __float2half_rn(uv.y);
    
  unsigned int uv_as_uint;
  pack2short(u, v, &uv_as_uint);
  return uv_as_uint;
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void unpack_uvwx ( unsigned int xy, unsigned int wx, float* uvw )
{
  unsigned short u_short;
  unsigned short v_short;
  unsigned short w_short;
  unsigned short x_short;

  unpack2short ( xy, &u_short, &v_short );
  unpack2short ( wx, &w_short, &x_short );

  uvw[0] = __half2float(u_short);
  uvw[1] = __half2float(v_short);
  uvw[2] = __half2float(w_short);
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
float2 unpack_uv ( unsigned int uv_as_int )
{
  unsigned short u_short;
  unsigned short v_short;

  unpack2short ( uv_as_int, &u_short, &v_short );
  
  float2 uv = float2_t ( __half2float(u_short), __half2float(v_short) );
  return uv;
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
uint4 intToUInt4 ( unsigned input )
{
  uint4 result;
  result.w = (input & 0xFF000000) >> 24U;
  result.z = (input & 0x00FF0000) >> 16U;
  result.y = (input & 0x0000FF00) >> 8U;
  result.x = (input & 0x000000FF);
  return result;
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
uint2 intToUInt2 ( unsigned input )
{
  uint2 result;
  result.y = (input & 0xFFFF0000) >> 16U;
  result.x = (input & 0x0000FFFF);
  return result;
}

__device__ inline
unsigned uint4ToUInt ( uint4 input )
{
  unsigned result = 0U;
  result |= (input.w & 0x000000FF) << 24U;
  result |= (input.z & 0x000000FF) << 16U;
  result |= (input.y & 0x000000FF) << 8U;
  result |= (input.x & 0x000000FF);
  return result;
}

__device__ inline
unsigned bvec4ToUInt ( uint4 input )
{
  unsigned result = 0U;
  result |= ((unsigned)(input.w) & 0x00000001) << 3U;
  result |= ((unsigned)(input.z) & 0x00000001) << 2U;
  result |= ((unsigned)(input.y) & 0x00000001) << 1U;
  result |= ((unsigned)(input.x) & 0x00000001);
  return result;
}

__device__ inline
uint4 uintToBvec4 ( unsigned input ) 
{
  uint4 result;
  result.w = (bool)((input & 0x00000008) >> 3U);
  result.z = (bool)((input & 0x00000004) >> 2U);
  result.y = (bool)((input & 0x00000002) >> 1U);
  result.x = (bool)((input & 0x00000001)      );
  return result;
}

#endif