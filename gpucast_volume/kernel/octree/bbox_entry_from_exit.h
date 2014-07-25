/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bbox_entry_from_exit.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_ENTRY_FROM_EXIT_H
#define LIBGPUCAST_CUDA_ENTRY_FROM_EXIT_H

#include <math/sign.h>

//////////////////////////////////////////////////////////////
template <typename source_t, typename target_t>
__device__
inline void bbox_entry_from_exit ( source_t const&  exit_point,
                                   source_t const&  exit_direction,
                                   float3 const&    bbox_min,
                                   float3 const&    bbox_max,
                                   target_t&        ray_entry,
                                   target_t&        ray_entry_normal )
{
  ray_entry_normal = 0.0f * ray_entry_normal;

  float tx = max((bbox_max.x - exit_point.x) / (-exit_direction.x), (bbox_min.x - exit_point.x) / (-exit_direction.x));
  float ty = max((bbox_max.y - exit_point.y) / (-exit_direction.y), (bbox_min.y - exit_point.y) / (-exit_direction.y));
  float tz = max((bbox_max.z - exit_point.z) / (-exit_direction.z), (bbox_min.z - exit_point.z) / (-exit_direction.z));

  if ( tx < ty && tx < tz ) 
  {
    ray_entry_normal.x  = sign (-exit_direction.x);
    ray_entry.x         = exit_point.x - tx * exit_direction.x;
    ray_entry.y         = exit_point.y - tx * exit_direction.y;
    ray_entry.z         = exit_point.z - tx * exit_direction.z;
    return;
  } else if ( ty < tx && ty < tz ) 
    {
      ray_entry_normal.y  = sign (-exit_direction.y);
      ray_entry.x         = exit_point.x - ty * exit_direction.x;
      ray_entry.y         = exit_point.y - ty * exit_direction.y;
      ray_entry.z         = exit_point.z - ty * exit_direction.z;
      return;
    } else if ( tz < tx && tz < ty ) 
      {
        ray_entry_normal.z  = sign (-exit_direction.z);
        ray_entry.x         = exit_point.x - tz * exit_direction.x;
        ray_entry.y         = exit_point.y - tz * exit_direction.y;
        ray_entry.z         = exit_point.z - tz * exit_direction.z;
        return;
      }
}

#endif // LIBGPUCAST_CUDA_ENTRY_FROM_EXIT_H