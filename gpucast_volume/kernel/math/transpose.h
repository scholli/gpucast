#ifndef LIB_GPUCAST_TRANSPOSE_H
#define LIB_GPUCAST_TRANSPOSE_H

/*********************************************************************
 * transpose of a 2x2 matrix
 *********************************************************************/
__device__
inline void
transpose2 ( float2* source, 
             float2* target ) 
{
  target[0] = float2_t(source[0].x, source[1].x);
  target[1] = float2_t(source[0].y, source[1].y);
}

/*********************************************************************
 * transpose of a 3x3 matrix
 *********************************************************************/
__device__
inline void
transpose3 ( float3* source, 
             float3* target ) 
{
  target[0] = float3_t(source[0].x, source[1].x, source[2].x);
  target[1] = float3_t(source[0].y, source[1].y, source[2].y);
  target[2] = float3_t(source[0].z, source[1].z, source[2].z);
}

#endif // LIB_GPUCAST_TRANSPOSE_H