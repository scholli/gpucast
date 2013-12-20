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
__kernel void evaluate_curve ( int              order, 
                               __global float4* points, 
                               __global float4* samples )
{
  int gti = get_group_id   (0);
  int ti  = get_local_id   (0);

  int n   = get_global_size (0);
  int nt  = get_local_size (0);

  float t = (float)(gti*nt+ti) / (n-1);

  int deg = order - 1;
  float u = 1.0 - t;
  
  float bc = 1.0;
  float tn = 1.0;

  float4 result = points[0] * u;

  for (int i = 1; i <= deg - 1; ++i) 
  {
    tn *= t;
    bc *= (float)(deg-i+1) / i;
    result = (result + tn * bc * points[i]) * u;
  } 
  result += tn * t * points[deg];
    
  // project into euclidian coordinates
  result.x = result.x/result.w;
  result.y = result.y/result.w;
  result.z = result.z/result.w;

  samples[gti*nt+ti] = result;
}

