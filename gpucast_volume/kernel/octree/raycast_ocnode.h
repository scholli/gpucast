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
#ifndef LIBGPUCAST_CUDA_RAYCAST_OCNODE_H
#define LIBGPUCAST_CUDA_RAYCAST_OCNODE_H

#include <math/raystate.h>

#include <octree/bubble_sort.h>
#include <octree/bbox_intersection.h>
#include <octree/raycast_faces.h>
#include <octree/bbox_intersect.h>
#include <octree/raycast_isosurface.h>

#include <gpucast/volume/isosurface/renderconfig.hpp>

// size of array for face intersections
// BE CAUTIOUS: octree has to be generated accordingly!
unsigned const MAX_ARRAY_SIZE_OF_FACES_PER_OCNODE = 128; 


///////////////////////////////////////////////////////////////////////////////
__device__ inline 
void find_face_intersections ( float4 const&      ray_entry,
                               float4 const&      ray_direction,
                               float              ray_exit_t,
                               float              isovalue,
                               unsigned           face_id,
                               unsigned           number_of_faces,
                               uint4 const*       facelistbuffer,
                               float4 const*      bboxbuffer,
                               float const*       limitbuffer,
                               bool               backface_culling,
                               bbox_intersection* intersections,
                               unsigned&          intersections_found )
{
  intersections_found = 0;

  for ( unsigned i = face_id; i != face_id + number_of_faces; ++i )
  {
    uint4 face             = facelistbuffer[i];
    float attrib_min       = limitbuffer[face.y];
    float attrib_max       = limitbuffer[face.y+1];
    bool contains_isovalue = isovalue >= attrib_min && isovalue <= attrib_max;
    bool is_outer_face     = face.w != 0;
 
    // intersect face only if it contains potential isovalue or is outer face
    if ( contains_isovalue || is_outer_face )
    {
      unsigned bbox_id   = face.z;
      float4 const* M    = bboxbuffer + bbox_id;
      float4 const* Minv = bboxbuffer + bbox_id + 4;
                     
      float3  low    = float3_t ( bboxbuffer[face.z+8].x, bboxbuffer[face.z+8].y, bboxbuffer[face.z+8].z);
      float3  high   = float3_t ( bboxbuffer[face.z+9].x, bboxbuffer[face.z+9].y, bboxbuffer[face.z+9].z);
      float4  center = bboxbuffer[face.z+10];

      float tmin = -1.0f; 
      float tmax = -1.0f;
      float2 uvmin, uvmax;

      if ( bbox_intersect ( ray_entry, ray_direction, M, Minv, low, high, center, backface_culling, tmin, tmax, uvmin, uvmax ) )
      {
        // add found intersection with front face
        intersections[intersections_found].surface_index  = face.x;
        intersections[intersections_found].uv             = uvmin;
        intersections[intersections_found].t              = tmin;
        ++intersections_found;

        // if backface culling is disabled -> also add backface
        if ( !backface_culling )        
        {
          intersections[intersections_found].surface_index  = face.x;
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
raycast_ocnode ( uint4 const&                 node, 
                 raystate&                    current_state,
                 uint4 const*                 facelistbuffer, 
                 float4 const*                bboxbuffer, 
                 float const*                 limitbuffer,
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
                 int&                         depth,
                 float3 const&                ocnode_min,
                 float3 const&                ocnode_max )
{
  unsigned const facelist_id = node.z;
  unsigned const faces       = node.w;
  
  // find all bbox intersections
  bbox_intersection intersections[MAX_ARRAY_SIZE_OF_FACES_PER_OCNODE]; // also use backfaces
  unsigned intersections_found = 0;
  
  // determine intersections with obb's of faces
  find_face_intersections ( ray_entry, 
                            ray_direction, 
                            ray_exit_t,
                            config.isovalue,
                            facelist_id, 
                            faces, 
                            facelistbuffer, 
                            bboxbuffer, 
                            limitbuffer, 
                            config.backface_culling, 
                            intersections, 
                            intersections_found );

  // sort obb intersections by depth along ray
  bubble_sort_bbox_intersections ( intersections, intersections_found );

  // intersect faces using obb's intersections as starting point
  raycast_faces ( config, surfacedatabuffer, surfacepointsbuffer, intersections, intersections_found, ray_entry, ray_direction, ray_exit_t, ocnode_min, ocnode_max );

  // sort face intersections
  bubble_sort_bbox_intersections ( intersections, intersections_found );
  
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

