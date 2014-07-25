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
#ifndef LIBGPUCAST_CUDA_BBOX_INTERSECT_H
#define LIBGPUCAST_CUDA_BBOX_INTERSECT_H

#include <octree/bbox_intersection.h>

#include <math/max.h>
#include <math/min.h>
#include <math/raystate.h>

///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool bbox_intersect ( float4 const& ray_origin,
                      float4 const& ray_direction,
                      float4 const* orientation,
                      float4 const* orientation_inv,
                      float3 const& low,
                      float3 const& high,
                      float4 const& center,
                      bool          backface_culling,
                      float&        t_min,
                      float&        t_max,
                      float2&       uv_min,
                      float2&       uv_max )
{
  float4 local_origin    = ray_origin - center;
  local_origin.w         = 1.0f;
  local_origin           = mult_mat4_float4 ( orientation_inv, local_origin );
  
  float4 local_direction = mult_mat4_float4 ( orientation_inv, float4_t(ray_direction.x, ray_direction.y, ray_direction.z, 0.0f) );

  float3 size            = high - low;

  float3 tlow            = (low  - float3_t ( local_origin.x, local_origin.y, local_origin.z )) / 
                                   float3_t ( local_direction.x, local_direction.y, local_direction.z );

  float3 thigh           = (high - float3_t ( local_origin.x, local_origin.y, local_origin.z )) / 
                                   float3_t ( local_direction.x, local_direction.y, local_direction.z );
  
  float3 tmin            = min ( tlow, thigh );
  float3 tmax            = max ( tlow, thigh );

  bool  min_intersect_found = false;

  // intersect with minimum planes
  float4 xmin_intersect = local_origin + tmin.x * local_direction;
  if ( xmin_intersect.y >= low.y && xmin_intersect.y <= high.y &&
       xmin_intersect.z >= low.z && xmin_intersect.z <= high.z )
  {
    uv_min              = float2_t ( (xmin_intersect.y - low.y) / size.y, (xmin_intersect.z - low.z) / size.z );
    t_min               = tmin.x;
    min_intersect_found = true;
  }

  float4 ymin_intersect = local_origin + tmin.y * local_direction;
  if ( ymin_intersect.x >= low.x && ymin_intersect.x <= high.x &&
       ymin_intersect.z >= low.z && ymin_intersect.z <= high.z )
  {
    uv_min              = float2_t ( (ymin_intersect.x - low.x) / size.x, (ymin_intersect.z - low.z) / size.z );
    t_min               = tmin.y;
    min_intersect_found = true;
  }

  float4 zmin_intersect = local_origin + tmin.z * local_direction;
  if ( zmin_intersect.x >= low.x && zmin_intersect.x <= high.x &&
       zmin_intersect.y >= low.y && zmin_intersect.y <= high.y )
  {
    uv_min              = float2_t ( (zmin_intersect.x - low.x) / size.x, (zmin_intersect.y - low.y) / size.y );
    t_min               = tmin.z;
    min_intersect_found = true;
  }
  
  uv_min = uv_min;

  // early exit if hit found 
  if ( backface_culling ) 
  {
    return min_intersect_found;
  }

  // intersect with maximum planes
  float4 xmax_intersect = local_origin + tmax.x * local_direction;
  if ( xmax_intersect.y >= low.y && xmax_intersect.y <= high.y &&
       xmax_intersect.z >= low.z && xmax_intersect.z <= high.z )
  {
    uv_max              = float2_t ( (xmax_intersect.y - low.y) / size.y, (xmax_intersect.z - low.z) / size.z );
    t_max               = tmax.x;
  }

  float4 ymax_intersect = local_origin + tmax.y * local_direction;
  if ( ymax_intersect.x >= low.x && ymax_intersect.x <= high.x &&
       ymax_intersect.z >= low.z && ymax_intersect.z <= high.z )
  {
    uv_max              = float2_t ( (ymax_intersect.x - low.x) / size.x, (ymax_intersect.z - low.z) / size.z );
    t_max               = tmax.y;
  }

  float4 zmax_intersect = local_origin + tmax.z * local_direction;
  if ( zmax_intersect.x >= low.x && zmax_intersect.x <= high.x &&
       zmax_intersect.y >= low.y && zmax_intersect.y <= high.y )
  {
    uv_max              = float2_t ( (zmax_intersect.x - low.x) / size.x, (zmax_intersect.y - low.y) / size.y );
    t_max               = tmax.z;
  }

  uv_max = uv_max;

  return min_intersect_found;// || max_intersect_found;
}

#endif // LIBGPUCAST_CUDA_BBOX_INTERSECT_H
