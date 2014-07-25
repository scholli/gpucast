/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : octree_lookup.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_OCTREE_LOOKUP_H
#define LIBGPUCAST_CUDA_OCTREE_LOOKUP_H

#include <math/floor.h>
#include <math/ceil.h>

//////////////////////////////////////////////////////////////
__device__
inline uint4 octree_lookup ( uint4 const*   tree,
                             float const*   limitbuffer,
                             int            index,
                             float3 const&  coords,
                             float3 const&  tree_min,
                             float3 const&  tree_max,
                             float          isovalue,
                             int&           depth,
                             float3&        ocnode_bbox_min,
                             float3&        ocnode_bbox_max )
{
  // set initial result
  uint4  node            = uint4_t(0, 0, 0, 0);
  depth                  = 0;

  float  local_coords[3] = { coords.x - tree_min.x, 
                             coords.y - tree_min.y, 
                             coords.z - tree_min.z };

  float3  tree_size      = tree_max - tree_min;

  float3  tree_center    = tree_size / 2.0f;

  bool is_inner_node  = (node.x <= 1);
  bool contains_outer = (node.x == 1 || node.x == 3);

  while ( is_inner_node ) // node is inner tree node
  {
    // read node
    node            = tree[index];
    is_inner_node   = (node.x <= 1);
    contains_outer  = (node.x == 1 || node.x == 3);

    if ( is_inner_node ) // if inner node
    {
      float attrib_min = limitbuffer[node.y  ];
      float attrib_max = limitbuffer[node.y+1];

      if ( (isovalue > attrib_min && isovalue < attrib_max) || // isovalue in attribute bounds of node or
            contains_outer )                                   // inner node including outer faces
      {
        // compute index of child node index
        index                 += 1 + int(local_coords[2] > tree_center.z);
        unsigned color_index   = int(local_coords[0] > tree_center.x) + 2 * int(local_coords[1] > tree_center.y);

        // fetch child node index
        uint4 tmp            = tree[index];
        unsigned children[4] = { tmp.x, tmp.y, tmp.z, tmp.w };
        index                = children[color_index]; 

        local_coords[0]  *= 2.0f;      
        local_coords[1]  *= 2.0f;
        local_coords[2]  *= 2.0f;

        if (local_coords[0] > tree_size.x) local_coords[0] -= tree_size.x;
        if (local_coords[1] > tree_size.y) local_coords[1] -= tree_size.y;
        if (local_coords[2] > tree_size.z) local_coords[2] -= tree_size.z;

        ++depth;
      } else { // attribute out of bounds and does not include outer faces -> return inner node
        break;
      }
    } 
  }

  // compute size of ocnode
  float scale             = pow(2.0f, float(depth));

  float3 relative_coords  = (coords - tree_min) / (tree_max - tree_min);
  ocnode_bbox_min         = floor (relative_coords * scale) / scale;
  ocnode_bbox_max         = ceil  (relative_coords * scale) / scale;
                     
  ocnode_bbox_min         = ocnode_bbox_min * tree_size;
  ocnode_bbox_max         = ocnode_bbox_max * tree_size;
                          
  ocnode_bbox_min         = ocnode_bbox_min + tree_min;
  ocnode_bbox_max         = ocnode_bbox_max + tree_min;

  return node;
}

#endif // LIBGPUCAST_CUDA_OCTREE_LOOKUP_H