/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_faces.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_RAYCAST_FACES_H
#define LIBGPUCAST_CUDA_RAYCAST_FACES_H

#include <octree/bbox_intersection.h>

#include <math/ray_to_plane_intersection.h>
#include <math/newton_surface.h>
#include <math/conversion.h>

#include <gpucast/volume/isosurface/renderconfig.hpp>

__device__ inline
void raycast_faces ( gpucast::renderconfig const& config,
                     uint4 const*                 surfacedatabuffer,
                     float4 const*                surfacepointsbuffer,
                     bbox_intersection*           intersections,
                     unsigned&                    intersections_found,
                     float4 const&                ray_origin,
                     float4 const&                ray_direction,
                     float                        ray_exit_t,
                     float3 const&                ocnode_min,
                     float3 const&                ocnode_max )
{
  // compute implitcit ray representation in form of two intersecting planes
  float d1  = 0.0f;
  float d2  = 0.0f;
  float3 n1 = float3_t(0.0f, 0.0f, 0.0f);
  float3 n2 = float3_t(0.0f, 0.0f, 0.0f);

  ray_to_plane_intersection ( ray_origin, ray_direction, &n1, &n2, &d1, &d2 );

  unsigned face_intersections = 0;

  for ( int i = 0; i != intersections_found; ++i ) 
  {
    unsigned surface_data_id    = intersections[i].surface_index;

    unsigned surface_points_id  = surfacedatabuffer[surface_data_id+1].y;
    uint2 order                 = uint2_t ( surfacedatabuffer[surface_data_id+2].y, 
                                            surfacedatabuffer[surface_data_id+2].z );

    float4 p, du, dv;
    unsigned newton_iterations = 0;

    bool is_intersection = newton_surface ( surfacepointsbuffer, 
                                            surface_points_id, 
                                            intersections[i].uv, 
                                            config.newton_epsilon, 
                                            config.newton_iterations, 
                                            order, 
                                            n1,
                                            n2,
                                            d1,
                                            d2,
                                            p,
                                            du,
                                            dv,
                                            newton_iterations );

    float t_intersection   = fabs(ray_direction.y) > fabs(ray_direction.x) ? (p.y - ray_origin.y) / ray_direction.y :  (p.x - ray_origin.x) / ray_direction.x;
    //bool hit_inside_ocnode = t_intersection >= 0.0f && t_intersection <= ray_exit_t;
    bool hit_inside_ocnode = (p.x >= ocnode_min.x && p.x <= ocnode_max.x) &&
                             (p.y >= ocnode_min.y && p.y <= ocnode_max.y) &&
                             (p.z >= ocnode_min.z && p.z <= ocnode_max.z);
    if ( is_intersection && 
         hit_inside_ocnode
       )
    {
      intersections[face_intersections]    = intersections[i];
      intersections[face_intersections].t  = t_intersection;
      ++face_intersections;
    } //else {
      // if there is no intersection in bounding box -> do not increment face_intersections
    //}
  }
  intersections_found = face_intersections;
}

#endif // LIBGPUCAST_CUDA_OCNODE_SIZE_H

