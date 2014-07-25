#ifndef LIB_GPUCAST_ADJUGATE_H
#define LIB_GPUCAST_ADJUGATE_H

#include "math/matrix_layout.h"

#ifdef COL_MAJOR

/*********************************************************************
 * adjugate a 2x2 matrix
 *********************************************************************/
__device__
inline void
adjugate2 ( float2* source, 
            float2* target ) 
{
  target[0].x =  source[1].y;
  target[0].y = -source[0].y;
  target[1].x = -source[1].x;
  target[1].y =  source[0].x;
}

/*********************************************************************
 * adjugate a 3x3 matrix
 *********************************************************************/
__device__
inline void
adjugate3 ( float3* source, 
            float3* target ) 
{
  target[0].x =   ( source[1].y*source[2].z - source[2].y*source[1].z );
  target[0].y = - ( source[0].y*source[2].z - source[2].y*source[0].z );
  target[0].z =   ( source[0].y*source[1].z - source[1].y*source[0].z );

  target[1].x = - ( source[1].x*source[2].z - source[2].x*source[1].z );
  target[1].y =   ( source[0].x*source[2].z - source[2].x*source[0].z );
  target[1].z = - ( source[0].x*source[1].z - source[1].x*source[0].z );

  target[2].x =   ( source[1].x*source[2].y - source[2].x*source[1].y );
  target[2].y = - ( source[0].x*source[2].y - source[2].x*source[0].y );
  target[2].z =   ( source[0].x*source[1].y - source[1].x*source[0].y );
}

#endif

#ifdef ROW_MAJOR

/*********************************************************************
 * adjugate a 2x2 matrix
 *********************************************************************/
__device__
void
adjugate2 ( float2* source, 
            float2* target ) 
{
  target[0].x =  source[1].y;
  target[0].y = -source[0].y;
  target[1].x = -source[1].x;
  target[1].y =  source[0].x;
}

/*********************************************************************
 * adjugate a 3x3 matrix
 *********************************************************************/
__device__
void
adjugate3 ( float3* source, 
            float3* target ) 
{
  target[0].x =   ( source[1].y*source[2].z - source[2].y*source[1].z );
  target[0].y = - ( source[1].x*source[2].z - source[1].z*source[2].x );
  target[0].z =   ( source[1].x*source[2].y - source[1].y*source[2].x );

  target[1].x = - ( source[0].y*source[2].z - source[0].z*source[2].y );
  target[1].y =   ( source[0].x*source[2].z - source[0].z*source[2].x );
  target[1].z = - ( source[0].x*source[2].y - source[0].y*source[2].x );

  target[2].x =   ( source[0].y*source[1].z - source[0].z*source[1].y );
  target[2].y = - ( source[0].x*source[1].z - source[0].z*source[1].x );
  target[2].z =   ( source[0].x*source[1].y - source[0].y*source[1].x );
}

#endif // ROW_MAJOR

#endif // LIB_GPUCAST_ADJUGATE_H