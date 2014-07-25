/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bbox_intersection.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_BBOX_INTERSECTION_H
#define LIBGPUCAST_CUDA_BBOX_INTERSECTION_H

struct bbox_intersection
{
  unsigned  surface_index;
  float2    uv;
  float     t;
};
#endif // LIBGPUCAST_CUDA_BBOX_INTERSECTION_H
