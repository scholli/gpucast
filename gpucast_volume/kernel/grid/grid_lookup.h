/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : grid_lookup.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_GRID_LOOKUP_H
#define LIBGPUCAST_CUDA_GRID_LOOKUP_H

#include <math/floor.h>
#include <math/ceil.h>
#include <math/in_domain.h>

#include <octree/bbox_exit_from_entry.h>

//////////////////////////////////////////////////////////////
__device__
inline uint4 grid_lookup ( uint4 const*   gridbuffer,
                           float3 const&  direction,
                           float3 const&  entry,
                           float3 const&  entry_normal,
                           int3 const&    grid_resolution,
                           float          epsilon,
                           float          isovalue,
                           float3&        exit,
                           float3&        exit_normal,
                           float&         ray_exit_t,
                           int3&          cell_coords,
                           bool&          gridcell_valid )
{
  float3 entry_with_offset = entry - epsilon * entry_normal;

  if ( !in_domain ( entry_with_offset, float3_t(0.0, 0.0, 0.0), float3_t(1.0, 1.0, 1.0) ) )
  {
    gridcell_valid = false;
    cell_coords    = int3_t(-1, -1, -1);
    exit           = float3_t(1.0, 0.0, 0.0);
    return uint4_t(0,0,0,0);
  } else {
    float3 cell_size     = float3_t(1.0f/grid_resolution.x, 1.0f/grid_resolution.y, 1.0f/grid_resolution.z);
     
    // compute cell coordinates using integer 
    cell_coords          = int3_t  ( int ( entry_with_offset.x / cell_size.x ),
                                     int ( entry_with_offset.y / cell_size.y ),
                                     int ( entry_with_offset.z / cell_size.z ) );
    
    float3 cell_bbox_min = cell_size * float3_t(cell_coords.x,   cell_coords.y,   cell_coords.z);
    float3 cell_bbox_max = cell_size * float3_t(cell_coords.x+1, cell_coords.y+1, cell_coords.z+1);

    // convert 3D-coords to 1D-index
    int cell_index = cell_coords.x + cell_coords.y * grid_resolution.x + cell_coords.z * grid_resolution.x * grid_resolution.y;

    // compute exit and normal 
    bbox_exit_from_entry ( entry, direction, cell_bbox_min, cell_bbox_max, exit, exit_normal, ray_exit_t);

    // return successful result
    gridcell_valid = true;
    return gridbuffer[cell_index];
  }
}

#endif // LIBGPUCAST_CUDA_OCTREE_LOOKUP_H