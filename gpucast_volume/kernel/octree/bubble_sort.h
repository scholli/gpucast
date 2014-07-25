/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bubble_sort.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_BUBBLE_SORT_H
#define LIBGPUCAST_CUDA_BUBBLE_SORT_H

#include <math/swap.h>
#include <octree/bbox_intersection.h>

__device__
inline void bubble_sort_bbox_intersections ( bbox_intersection* intersections,
                                             unsigned total_intersections )
{
  // sort list of fragments
  for ( int n = total_intersections-1; n > 0; --n ) 
  {
    for ( int i = 0; i != n; ++i ) 
    {
      if ( intersections[i].t > intersections[i+1].t ) 
      {
        swap ( intersections[i], intersections[i+1] );
      }
    }
  }
}

#endif // LIBGPUCAST_CUDA_BUBBLE_SORT_H
