#ifndef LIBGPUCAST_POINT_RAY_DISTANCE_H
#define LIBGPUCAST_POINT_RAY_DISTANCE_H

/////////////////////////////////////////////////////////////////////////////
__device__ inline
float point_ray_distance ( float3 const& ray_origin,
                           float3 const& ray_direction,
                           float3 const& point )
{
  float t0  = dot ( ray_direction, (point - ray_origin) ) / dot(ray_direction, ray_direction);
  float d   = length( point - (ray_origin + t0 * ray_direction) );
  return d;
}

#endif // LIBGPUCAST_POINT_RAY_DISTANCE_H