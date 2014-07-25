#ifndef LIB_GPUCAST_ADJUGATE_H
#define LIB_GPUCAST_ADJUGATE_H

/*********************************************************************
 * adjugate a 2x2 matrix
 *********************************************************************/
__device__
void
adjugate ( float2 source[2], 
           float2 target[2] ) 
{
  target[0].x =  source[1].y;
  target[0].y = -source[0].y;
  target[1].x = -source[1].x;
  target[1].y =  source[0].x;
}

/*********************************************************************
 * adjugate a 2x2 matrix
 *********************************************************************/
__device__
void
adjugate ( float3 source[3], 
           float3 target[3] ) 
{
  target[0].s0 = source[1].s1 * source[2].s2 - source[]
}

#endif // LIB_GPUCAST_ADJUGATE_H