/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : evaluate_curve.cl
*  project    : glpp
*  description:
*
********************************************************************************/
#define BLOCKSIZE       32

float4 evaluate_curve ( int              order,
                        float            t,
                        __local float4   local_points[BLOCKSIZE] )
{
  float u = 1.0 - t;
  
  float bc = 1.0;
  float tn = 1.0;

  float4 result = local_points[0] * u;

  for (int i = 1; i <= order - 2; ++i) 
  {
    tn *= t;
    bc *= (float)(order-i) / i;
    result = (result + tn * bc * local_points[i]) * u;
  } 
  result += tn * t * local_points[order-1];
    
  /* project into euclidian coordinates */
  result.x = result.x/result.w;
  result.y = result.y/result.w;
  result.z = result.z/result.w;
  
  return result;
}


__kernel void run_kernel ( __global float4* points, 
                           __global float4* samples,
                           int              order)
{
  __local float4 local_points[BLOCKSIZE]; 

  int gti = get_group_id    (0);
  int ti  = get_local_id    (0);

  int n   = get_global_size (0);
  int nt  = get_local_size  (0);

  // for each block copy points into local array
  local_points[ti] = points[ti];
  barrier(CLK_LOCAL_MEM_FENCE); 

  float t = (float)(gti*nt+ti) / (n-1);
  samples[gti*nt+ti] = evaluate_curve ( order, t, local_points );
  barrier(CLK_LOCAL_MEM_FENCE); 
}
