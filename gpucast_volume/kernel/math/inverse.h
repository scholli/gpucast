#ifndef LIB_GPUCAST_INVERSE_H
#define LIB_GPUCAST_INVERSE_H

#include "math\adjugate.h"
#include "math\determinant.h"

/*********************************************************************
 * inverse of a 2x2 matrix
 *********************************************************************/
__device__
inline void
inverse2 ( float2* source, 
           float2* target ) 
{
  float   det = determinant2(source);
  float2  adj[2];

  adjugate2(source, adj);

  target[0] = adj[0]/det;
  target[1] = adj[1]/det; 
}

/*********************************************************************
 * inverse of a 2x2 matrix
 *********************************************************************/
__device__
inline void
inverse3 ( float3* source, 
           float3* target ) 
{
  float   det = determinant3(source);
  float3  adj[3];

  adjugate3(source, adj);

  target[0] = adj[0]/det;
  target[1] = adj[1]/det;
  target[2] = adj[2]/det;
}

#endif // LIB_GPUCAST_INVERSE_H