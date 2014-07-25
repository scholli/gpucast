/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raystate.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_RAYSTATE_H
#define LIBGPUCAST_CUDA_RAYSTATE_H

struct raystate
{
  unsigned volume;
  unsigned adjacent_volume;
  unsigned surface;
  float    depth;
  float3   uvw;
  float4   point;
};

#endif // LIBGPUCAST_CUDA_RAYSTATE_H
