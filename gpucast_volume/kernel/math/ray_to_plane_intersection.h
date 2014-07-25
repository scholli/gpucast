#ifndef LIB_GPUCAST_RAY_TO_PLANE_INTERSECTION_H
#define LIB_GPUCAST_RAY_TO_PLANE_INTERSECTION_H

#include <math/length.h>
#include <math/dot.h>
#include <math/cross.h>
#include <math/normalize.h>

/**********************************************************************
* generate ray defined by intersecting planes
***********************************************************************/
__device__
inline void 
point_to_plane_intersection ( float4 const& p_objectspace,
                              float4 const* mv_inv,
                              float3*       n1, 
                              float3*       n2, 
                              float*        d1, 
                              float*        d2 )
{
  float4 rayorigin = float4_t ( mv_inv[3].x, mv_inv[3].y, mv_inv[3].z, 1.0f);
  float3 r         = normalize ( float3_t ( p_objectspace.x, p_objectspace.y, p_objectspace.z ) - float3_t ( rayorigin.x, rayorigin.y, rayorigin.z ) );
  float4 raydir    = float4_t ( r.x, r.y, r.z, 0.0f );

  if ( fabs(raydir.x) > fabs(raydir.y) && 
       fabs(raydir.x) > fabs(raydir.z) ) 
  {
    *n1 = float3_t(raydir.y, -raydir.x, 0.0);
  } else {
    *n1 = float3_t(0.0, raydir.z, -raydir.y);
  }

  *n2 = cross(*n1, float3_t(raydir.x, raydir.y, raydir.z));

  *n1 = normalize(*n1);
  *n2 = normalize(*n2);

  *d1 = dot(-1.0f * (*n1), float3_t(rayorigin.x, rayorigin.y, rayorigin.z));
  *d2 = dot(-1.0f * (*n2), float3_t(rayorigin.x, rayorigin.y, rayorigin.z));
}


/////////////////////////////////////////////////////////////////////
__device__
inline void 
ray_to_plane_intersection ( float4 const& ray_origin,
                            float4 const& ray_direction,
                            float3*       n1,
                            float3*       n2,
                            float*        d1,
                            float*        d2 )
{
  if ( fabs(ray_direction.x) > fabs(ray_direction.y)  && 
       fabs(ray_direction.x) > fabs(ray_direction.z) ) 
  {
    *n1 = float3_t(ray_direction.y, -ray_direction.x, 0.0f);
  } else {
    *n1 = float3_t(0.0f, ray_direction.z, -ray_direction.y);
  }

  *n2 = cross(*n1, float3_t(ray_direction.x,ray_direction.y,ray_direction.z));

  *n1 = normalize(*n1);
  *n2 = normalize(*n2);

  *d1 = dot(-1.0f * (*n1), float3_t(ray_origin.x,ray_origin.y,ray_origin.z));
  *d2 = dot(-1.0f * (*n2), float3_t(ray_origin.x,ray_origin.y,ray_origin.z));
}

#endif // LIB_GPUCAST_RAY_TO_PLANE_INTERSECTION_H
