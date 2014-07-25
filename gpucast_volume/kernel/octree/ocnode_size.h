/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : ocnode_size.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_OCNODE_SIZE_H
#define LIBGPUCAST_CUDA_OCNODE_SIZE_H

#include <math/floor.h>
#include <math/ceil.h>

//////////////////////////////////////////////////////////////
__device__
inline void ocnode_size ( float3  sample_point,
                          float3  octree_min,
                          float3  octree_max,
                          int     depth,
                          float3* ocnode_bbox_min,
                          float3* ocnode_bbox_max )
{
  float3 tree_size     = octree_max - octree_min;

  float scale          = pow(2.0f, float(depth));
  float3 local_coords  = (sample_point - octree_min) / tree_size;
  *ocnode_bbox_min     = floor (local_coords * scale) / scale;
  *ocnode_bbox_max     = ceil  (local_coords * scale) / scale;
                      
  *ocnode_bbox_min     = (*ocnode_bbox_min) * tree_size;
  *ocnode_bbox_max     = (*ocnode_bbox_max) * tree_size;
                       
  *ocnode_bbox_min     = (*ocnode_bbox_min) + octree_min;
  *ocnode_bbox_max     = (*ocnode_bbox_max) + octree_min;
}

#endif // LIBGPUCAST_CUDA_OCNODE_SIZE_H
