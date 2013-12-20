/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : point_ray_distance.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_POINT_RAY_DISTANCE_FRAG
#define LIBGPUCAST_POINT_RAY_DISTANCE_FRAG

/////////////////////////////////////////////////////////////////////////////
float point_ray_distance ( in vec3   ray_origin,
                           in vec3   ray_direction,
                           in vec3   point )
{
  float t0  = dot ( ray_direction, (point - ray_origin) ) / dot(ray_direction, ray_direction);
  float d   = length( point - (ray_origin + t0 * ray_direction) );
  return d;
}

#endif