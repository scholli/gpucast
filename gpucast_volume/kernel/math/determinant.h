#ifndef LIB_GPUCAST_DETERMINANT_H
#define LIB_GPUCAST_DETERMINANT_H

#include "math/matrix_layout.h"

#ifdef COL_MAJOR

/*********************************************************************
 * determinant of a 2x2 matrix
 *********************************************************************/
__device__
inline float
determinant2 ( float2* m ) 
{
  return m[0].x * m[1].y - 
         m[0].y * m[1].x;
}

/*********************************************************************
 * determinant of a 3x3 matrix
 *********************************************************************/
__device__
inline float
determinant3 ( float3* m ) 
{
  return m[0].x * m[1].y * m[2].z +
         m[1].x * m[2].y * m[0].z +
         m[2].x * m[0].y * m[1].z -
         m[0].x * m[2].y * m[1].z -
         m[1].x * m[0].y * m[2].z -
         m[2].x * m[1].y * m[0].z;
}

#endif

#ifdef ROW_MAJOR

/*********************************************************************
 * determinant of a 2x2 matrix
 *********************************************************************/
__device__
float
determinant2 ( float2* m ) 
{
  return m[0].x * m[1].y - 
         m[1].x * m[0].y;
}

/*********************************************************************
 * determinant of a 3x3 matrix
 *********************************************************************/
__device__
float
determinant3 ( float3* m ) 
{
  return m[0].x * m[1].y * m[2].z +
         m[0].y * m[1].z * m[2].x +
         m[0].z * m[1].x * m[2].y -
         m[0].x * m[1].z * m[2].y -
         m[0].y * m[1].x * m[2].z -
         m[0].z * m[1].y * m[2].x;
}

#endif

#endif // LIB_GPUCAST_H_DETERMINANT_H