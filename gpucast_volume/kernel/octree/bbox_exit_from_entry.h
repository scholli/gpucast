/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : bbox_exit_from_entry.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_EXIT_FROM_ENTRY_H
#define LIBGPUCAST_CUDA_EXIT_FROM_ENTRY_H

//////////////////////////////////////////////////////////////
__device__
inline void bbox_exit_from_entry ( float4 const&  entry_point,
                                   float4 const&  entry_direction,
                                   float3 const&  bbox_min,
                                   float3 const&  bbox_max,
                                   float4&        ray_exit,
                                   float4&        ray_exit_normal,
                                   float&         ray_exit_t )
{
  ray_exit_normal = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  float tx        = 0.0f;
  float ty        = 0.0f;
  float tz        = 0.0f;

  tx = max((bbox_max.x - entry_point.x) / entry_direction.x, (bbox_min.x - entry_point.x) / entry_direction.x);
  ty = max((bbox_max.y - entry_point.y) / entry_direction.y, (bbox_min.y - entry_point.y) / entry_direction.y);
  tz = max((bbox_max.z - entry_point.z) / entry_direction.z, (bbox_min.z - entry_point.z) / entry_direction.z);

  if ( tx <= ty && tx <= tz ) 
  {
    ray_exit_normal.x = sign(entry_direction.x);
    ray_exit          = entry_point + tx * entry_direction;
    ray_exit_t        = tx;
    return;
  } else if ( ty <= tx && ty <= tz ) 
    {
      ray_exit_normal.y = sign(entry_direction.y);
      ray_exit          = entry_point + ty * entry_direction;
      ray_exit_t        = ty;
      return;
    } else if ( tz <= tx && tz <= ty ) 
      {
        ray_exit_normal.z = sign(entry_direction.z);
        ray_exit          = entry_point + tz * entry_direction;
        ray_exit_t        = tz;
        return;
      }
}

//////////////////////////////////////////////////////////////
__device__
inline void bbox_exit_from_entry ( float3 const&  entry_point,
                                   float3 const&  entry_direction,
                                   float3 const&  bbox_min,
                                   float3 const&  bbox_max,
                                   float3&        ray_exit,
                                   float3&        ray_exit_normal,
                                   float&         ray_exit_t )
{
  ray_exit_normal = float3_t(0.0f, 0.0f, 0.0f);
  float tx        = 0.0f;
  float ty        = 0.0f;
  float tz        = 0.0f;

  tx = max((bbox_max.x - entry_point.x) / entry_direction.x, (bbox_min.x - entry_point.x) / entry_direction.x);
  ty = max((bbox_max.y - entry_point.y) / entry_direction.y, (bbox_min.y - entry_point.y) / entry_direction.y);
  tz = max((bbox_max.z - entry_point.z) / entry_direction.z, (bbox_min.z - entry_point.z) / entry_direction.z);

  if ( tx <= ty && tx <= tz ) 
  {
    ray_exit_normal.x = sign(entry_direction.x);
    ray_exit          = entry_point + tx * entry_direction;
    ray_exit_t        = tx;
    return;
  } else if ( ty <= tx && ty <= tz ) 
    {
      ray_exit_normal.y = sign(entry_direction.y);
      ray_exit          = entry_point + ty * entry_direction;
      ray_exit_t        = ty;
      return;
    } else if ( tz <= tx && tz <= ty ) 
      {
        ray_exit_normal.z = sign(entry_direction.z);
        ray_exit          = entry_point + tz * entry_direction;
        ray_exit_t        = tz;
        return;
      }
}

#endif // LIBGPUCAST_CUDA_EXIT_FROM_ENTRY_H