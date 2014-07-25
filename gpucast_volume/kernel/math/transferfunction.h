/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : transferfunction.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_TRANSFERFUNCTION_H
#define LIBGPUCAST_TRANSFERFUNCTION_H

__device__ inline
float4 transferfunction ( float normalized )
{
  // clamp
  if ( normalized < 0.0 ) normalized = 0.0;
  if ( normalized > 1.0 ) normalized = 1.0;

  float limit0 = 0.0;  // val0
  float limit1 = 0.1; // val1
  float limit2 = 0.2;  // val2
  float limit3 = 0.5; // val3
  float limit4 = 1.0;  // val4

  float4 val0 = float4_t(0.0, 0.0, 0.5, 1.0);
  float4 val1 = float4_t(0.0, 0.7, 0.5, 1.0);
  float4 val2 = float4_t(0.0, 0.9, 0.0, 1.0);
  float4 val3 = float4_t(0.9, 0.9, 0.0, 1.0);
  float4 val4 = float4_t(0.9, 0.0, 0.0, 1.0);

  if ( normalized >= limit0 && normalized <= limit1 ) 
  {
    float span     = limit1 - limit0;
    float relative = (limit1 - normalized) / span;
    return mix ( val1, val0, relative );
  }
  
  if ( normalized >= limit1 && normalized <= limit2 ) 
  {
    float span     = limit2 - limit1;
    float relative = (limit2 - normalized) / span;
    return mix ( val2, val1, relative );
  }

  if ( normalized >= limit2 && normalized <= limit3 ) 
  {
    float span     = limit3 - limit2;
    float relative = (limit3 - normalized) / span;
    return mix ( val3, val2, relative );
  }

  if ( normalized >= limit3 && normalized <= limit4 ) 
  {
    float span     = limit4 - limit3;
    float relative = (limit4 - normalized) / span;
    return mix ( val4, val3, relative );
  }

  return float4_t(0.0, 0.0, 0.0, 1.0);
}

#endif

