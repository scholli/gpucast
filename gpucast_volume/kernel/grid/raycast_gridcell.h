/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_gridcell.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_RAYCAST_GRIDCELL_H
#define LIBGPUCAST_CUDA_RAYCAST_GRIDCELL_H

#include <math/raystate.h>

#include <octree/bubble_sort.h>
#include <octree/bbox_intersection.h>
#include <octree/raycast_faces.h>
#include <octree/bbox_intersect.h>
#include <octree/raycast_isosurface.h>

#include <gpucast/volume/isosurface/renderconfig.hpp>

// size of array for face intersections
// BE CAUTIOUS: octree has to be generated accordingly!
unsigned const MAX_ARRAY_SIZE_OF_FACES_PER_GRIDCELL = 128; 


///////////////////////////////////////////////////////////////////////////////
__device__ inline 
void find_face_intersections ( float4 const&      ray_entry,
                               float4 const&      ray_direction,
                               float              ray_exit_t,
                               float              isovalue,
                               unsigned           face_id,
                               unsigned           number_of_faces,
                               uint4 const*       facebuffer,
                               unsigned           facebuffersize,
                               float4 const*      bboxbuffer,
                               uint4 const*       surfacedatabuffer,
                               bool               backface_culling,
                               bbox_intersection* intersections,
                               unsigned&          intersections_found )
{
  intersections_found = 0;

  for ( unsigned i = face_id; i != face_id + number_of_faces; ++i )
  {
    if ( i >= facebuffersize ) return;

    uint4 face               = facebuffer[i];

    unsigned surface_data_id = face.z;
    unsigned bbox_id         = face.x;
    bool is_outer_face       = face.y != 0;

    float attrib_min         = intBitsToFloat(surfacedatabuffer[surface_data_id].x);
    float attrib_max         = intBitsToFloat(surfacedatabuffer[surface_data_id].y);
    bool contains_isovalue   = isovalue >= attrib_min && isovalue <= attrib_max;
    
    // intersect face only if it contains potential isovalue or is outer face
    if ( contains_isovalue || is_outer_face )
    {
      float4 const* M    = bboxbuffer + bbox_id;
      float4 const* Minv = bboxbuffer + bbox_id + 4;
                     
      float3  low    = float3_t ( bboxbuffer[bbox_id+8].x, bboxbuffer[bbox_id+8].y, bboxbuffer[bbox_id+8].z);
      float3  high   = float3_t ( bboxbuffer[bbox_id+9].x, bboxbuffer[bbox_id+9].y, bboxbuffer[bbox_id+9].z);
      float4  center = bboxbuffer[bbox_id+10];

      float tmin = -1.0f; 
      float tmax = -1.0f;
      float2 uvmin, uvmax;

      if ( bbox_intersect ( ray_entry, ray_direction, M, Minv, low, high, center, backface_culling, tmin, tmax, uvmin, uvmax ) )
      {
        // add found intersection with front face
        intersections[intersections_found].surface_index  = surface_data_id;
        intersections[intersections_found].uv             = uvmin;
        intersections[intersections_found].t              = tmin;
        ++intersections_found;

        // if backface culling is disabled -> also add backface
        if ( !backface_culling )        
        {
          intersections[intersections_found].surface_index  = surface_data_id;
          intersections[intersections_found].uv             = uvmax;
          intersections[intersections_found].t              = tmax;
          ++intersections_found;
        }
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline void
raycast_gridcell ( uint4 const&                 gridcell, 
                   raystate&                    current_state,
                   uint4 const*                 facebuffer, 
                   unsigned                     facebuffersize,
                   float4 const*                bboxbuffer, 
                   uint4 const*                 surfacedatabuffer,
                   float4 const*                surfacepointsbuffer,
                   float4 const*                volumedatabuffer,
                   float4 const*                volumepointsbuffer,
                   float4 const*                attributedatabuffer,
                   float2 const*                attributepointsbuffer,
                   float4 const&                ray_entry, 
                   float4 const&                ray_exit, 
                   float                        ray_exit_t,
                   float4 const&                ray_direction, 
                   gpucast::renderconfig const& config,
                   gpucast::bufferinfo const&   info,
                   float4 const*                modelview,
                   float4 const*                normalmatrix,
                   float4 const&                external_color,
                   float const&                 external_depth,
                   float&                       out_depth,
                   float4&                      out_color,
                   float&                       out_opacity,
                   float3 const&                gridcell_min,
                   float3 const&                gridcell_max )
{
  unsigned const facelist_id = gridcell.w;
  uint2 tmp                  = intToUInt2(gridcell.z);
  unsigned faces             = tmp.x;
  
  // find all bbox intersections
  bbox_intersection intersections[MAX_ARRAY_SIZE_OF_FACES_PER_GRIDCELL]; // also use backfaces
  unsigned intersections_found = 0;

  // determine intersections with obb's of faces
  find_face_intersections ( ray_entry, 
                            ray_direction, 
                            ray_exit_t,
                            config.isovalue,
                            facelist_id, 
                            faces, 
                            facebuffer, 
                            facebuffersize,
                            bboxbuffer, 
                            surfacedatabuffer, 
                            config.backface_culling, 
                            intersections, 
                            intersections_found );

  // sort obb intersections by depth along ray
  bubble_sort_bbox_intersections ( intersections, intersections_found );

  // intersect faces using obb's intersections as starting point
  raycast_faces ( config, surfacedatabuffer, surfacepointsbuffer, intersections, intersections_found, ray_entry, ray_direction, ray_exit_t, gridcell_min, gridcell_max );

  // sort face intersections
  bubble_sort_bbox_intersections ( intersections, intersections_found );
#if 0
  if ( intersections_found > 0 )
  {
    out_color = out_color + out_opacity * float4_t(1.0, 0.0, 0.0, 1.0);
    out_depth = 0.3;
    out_opacity *= 0.6;
    return;
  } else {
    out_color = out_color + out_opacity * float4_t(1.0, 0.0, 1.0, 1.0);
    out_depth = 0.3;
    out_opacity *= 0.6;
    return;
  }
#endif
  // if there are intersections with faces -> find 
  raycast_isosurface ( config,
                       info,
                       modelview,
                       normalmatrix,
                       surfacedatabuffer,
                       surfacepointsbuffer,
                       volumedatabuffer,
                       volumepointsbuffer,
                       attributedatabuffer,
                       attributepointsbuffer,
                       intersections, 
                       intersections_found,
                       external_color,
                       external_depth,
                       out_depth,
                       out_color,
                       out_opacity,
                       current_state,
                       ray_entry,
                       ray_exit,
                       ray_exit_t,
                       ray_direction );
}


#endif // LIBGPUCAST_CUDA_OCNODE_SIZE_H

