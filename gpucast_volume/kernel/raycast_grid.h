/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_grid.h
*  project    : gpucast
*  description:
*
********************************************************************************/

#include <math/mult.h>
#include <math/length.h>
#include <math/operator.h>

#include <octree/bbox_entry_from_exit.h>
#include <octree/bbox_exit_from_entry.h>

#include <grid/grid_lookup.h>
#include <grid/raycast_gridcell.h>

#include <gpucast/volume/isosurface/renderconfig.hpp>

// just for numerical robustness -> abort after max. iterations
// practically NEVER happens!
unsigned const MAX_OCNODE_INTERSECTIONS = 1024;

///////////////////////////////////////////////////////////////////////////////
__device__
void raycast_grid ( gpucast::renderconfig const&  config,
                    gpucast::bufferinfo           info,
                    int2                          coords,
                    int3                          grid_resolution,
                    float4 const*                 matrixbuffer,
                    uint4 const*                  gridbuffer,
                    uint4 const*                  facebuffer,
                    float4 const*                 bboxbuffer,
                    uint4 const*                  surfacedatabuffer,
                    float4 const*                 surfacepointsbuffer,
                    float4 const*                 volumedatabuffer,
                    float4 const*                 volumepointsbuffer,
                    float4 const*                 attributedatabuffer,
                    float2 const*                 attributepointsbuffer,
                    __write_only image2d_t        colortexture,
                    __write_only image2d_t        depthtexture,
                    __read_only image2d_t         input_position_image,
                    __read_only image2d_t         external_image )
{

  /*************************************
  * get external fbo and inforamtion for current fragment
  **************************************/
  float4 external_color_depth;
  surf2Dread ( &external_color_depth, external_image, coords.x*sizeof(float4), coords.y );
  float4 external_color = float4_t (external_color_depth.x, external_color_depth.y, external_color_depth.z, 1.0f);
  float  external_depth = external_color_depth.w;

  float4 input_position;
  surf2Dread ( &input_position, input_position_image, coords.x*sizeof(float4), coords.y );

  /*************************************
  * no fragments -> just use external color/depth if available
  **************************************/
  if ( input_position.w < 0.5f ) 
  {
    if ( external_depth < 1.0f ) 
    {
      surf2Dwrite ( external_color, colortexture, coords.x*sizeof(float4), coords.y );
      surf2Dwrite ( external_depth, depthtexture, coords.x*sizeof(float), coords.y );
    }
    return;
  } 

  /*************************************
  * get transformation matrices
  **************************************/
  float4 const*   modelview                  = matrixbuffer + 0;
  //float4 const*   modelviewprojection        = matrixbuffer + 4;
  float4 const*   modelviewinverse           = matrixbuffer + 8;
  float4 const*   normalmatrix               = matrixbuffer + 12;
  //float4 const*   modelviewprojectioninverse = matrixbuffer + 16;

  /*************************************
  * ray setup
  **************************************/
  float4 const   ray_origin          = modelviewinverse * float4_t(0.0f, 0.0f, 0.0f, 1.0f);
  float4 const   ray_direction       = normalize ( float4_t ( input_position.x - ray_origin.x, 
                                                              input_position.y - ray_origin.y, 
                                                              input_position.z - ray_origin.z, 0.0f ) ); 

  float3 const   grid_min            = float3_t ( config.bbox_min[0], config.bbox_min[1], config.bbox_min[2] );
  float3 const   grid_max            = float3_t ( config.bbox_max[0], config.bbox_max[1], config.bbox_max[2] );
  float3 const   grid_size           = grid_max - grid_min;
  
  float3 entry_object_space;
  float3 entry_normal;
  bbox_entry_from_exit ( input_position, ray_direction, grid_min, grid_max, entry_object_space, entry_normal );

  float3          gridcell_size      = float3_t(1.0f/grid_resolution.x, 1.0f/grid_resolution.y, 1.0f/grid_resolution.z);
  unsigned const  max_grid_dimension = max ( grid_resolution.x, max ( grid_resolution.y, grid_resolution.z ) );
  float const     epsilon_offset     = 1.0f / (2*max_grid_dimension); // offset is half of minimal cell size

  float3 const exit_grid_coordinates  = float3_t( ( input_position.x - grid_min.x ) / grid_size.x,
                                                  ( input_position.y - grid_min.y ) / grid_size.y,
                                                  ( input_position.z - grid_min.z ) / grid_size.z );

  float3 const entry_grid_coordinates = float3_t( ( entry_object_space.x - grid_min.x ) / grid_size.x,
                                                  ( entry_object_space.y - grid_min.y ) / grid_size.y,
                                                  ( entry_object_space.z - grid_min.z ) / grid_size.z );

  float distance_to_origin = length ( entry_grid_coordinates );

  float3 gridcell_entry         = entry_grid_coordinates;
  float3 gridcell_entry_normal  = entry_normal;

  float3 ray_direction_grid_coordinates = exit_grid_coordinates - entry_grid_coordinates;
  ray_direction_grid_coordinates = normalize(ray_direction_grid_coordinates);

  float3 gridcell_exit_normal;
  float3 gridcell_exit;  

  /*************************************
  * state setup for ray casting
  **************************************/
  bool continue_raytracing           = true;
  unsigned iterations                = 0;

  float  out_depth                   = 0.0f;
  float4 out_color                   = float4_t(0.0, 0.0, 0.0, 0.0);
  float  out_opacity                 = 1.0f;
  float const out_saturation_reached = 0.01f;

  raystate current_state;
  current_state.volume  = 0;
  current_state.surface = 0;
  current_state.uvw     = float3_t(0.0f, 0.0f, 0.0f);
  current_state.depth   = 0.0f;

  unsigned old_cell_index = 0xFFFF;

  /*************************************
  * raycast octree representation
  **************************************/
  while ( continue_raytracing )
  {
    /*************************************
    * get gridcell for current ray position
    **************************************/
    bool gridcell_valid = false;
    float gridcell_exit_t;

    int3  gridcell_coords;
    uint4 gridcell  = grid_lookup ( gridbuffer,
                                    ray_direction_grid_coordinates,
                                    gridcell_entry,
                                    gridcell_entry_normal,
                                    grid_resolution,
                                    epsilon_offset,
                                    config.isovalue,
                                    gridcell_exit,
                                    gridcell_exit_normal,
                                    gridcell_exit_t,
                                    gridcell_coords,
                                    gridcell_valid );

    /*************************************
    * critical abort -> same cell hit twice
    **************************************/
    if ( old_cell_index == gridcell.w ) { 
      continue_raytracing = false;
      //out_color = float4_t(1.0, 0.0, 0.0, 1.0);
    } else {
      old_cell_index = gridcell.w;
    }

    /*************************************
    * if inner node -> no leafs to examine -> skip node
    **************************************/
    if ( gridcell_valid ) // if inner node hit -> intersect inner node and continue at exit point
    { // if leaf node hit
      /*************************************
      * if contains outer surface or face with according attribute bounds -> examine node
      **************************************/
      float attrib_min         = intBitsToFloat(gridcell.x);
      float attrib_max         = intBitsToFloat(gridcell.y);
      bool contains_isovalue   = (config.isovalue > attrib_min) && (config.isovalue < attrib_max);

      out_color    = out_color + 0.001 * float4_t (0.0, 1.0, 0.0, 1.0);
      out_opacity *= 0.999;

      uint2 tmp                = intToUInt2(gridcell.z);
      unsigned nfaces          = tmp.x;
      bool contains_outer_face = tmp.y != 0;    

      if ( contains_isovalue || contains_outer_face ) // if node does contain potential faces
      {
        float4 ray_entry = float4_t(grid_min.x + gridcell_entry.x * grid_size.x,
                                    grid_min.y + gridcell_entry.y * grid_size.y,
                                    grid_min.z + gridcell_entry.z * grid_size.z,
                                    1.0f);

        float4 ray_exit  = float4_t(grid_min.x + gridcell_exit.x  * grid_size.x,
                                    grid_min.y + gridcell_exit.y  * grid_size.y,
                                    grid_min.z + gridcell_exit.z  * grid_size.z,
                                    1.0f);

        float3 gridcell_min = grid_min + gridcell_size * float3_t(gridcell_coords.x,   gridcell_coords.y,   gridcell_coords.z  ) * grid_size;
        float3 gridcell_max = grid_min + gridcell_size * float3_t(gridcell_coords.x+1, gridcell_coords.y+1, gridcell_coords.z+1) * grid_size;

        // iterate face list
        raycast_gridcell ( gridcell,
                           current_state,
                           facebuffer, 
                           info.facebuffer_size,
                           bboxbuffer, 
                           surfacedatabuffer,
                           surfacepointsbuffer,
                           volumedatabuffer,
                           volumepointsbuffer,
                           attributedatabuffer,
                           attributepointsbuffer,
                           ray_entry,
                           ray_exit,
                           gridcell_exit_t,
                           ray_direction, 
                           config,
                           info,
                           modelview,
                           normalmatrix,
                           external_color,
                           external_depth,
                           out_depth,
                           out_color,
                           out_opacity,
                           gridcell_min,
                           gridcell_max );
      } else { 
        // if node does NOT contain potential faces
      }
    } else {
      continue_raytracing = false;
    }

    /*************************************
    * reached external opaque surface -> end ray casting 
    * external color has been applied in raycast_ocnode()
    **************************************/
    if ( out_depth > external_depth )
    {
      continue_raytracing = false;
    }

    if ( out_opacity < 0.01 )
    {
      continue_raytracing = false;
    }

    gridcell_entry        = gridcell_exit;
    gridcell_entry_normal = -1.0f * gridcell_exit_normal;
  }

  /********************************************************************************
  * volume ray casting ended -> check if there is an external hit behind current ray position
  ********************************************************************************/   
  if ( external_depth < 1.0f ) 
  {
    out_depth = external_depth;
    out_color = out_color + out_opacity * external_color;
  }

  /*************************************
  * write result to fbo
  **************************************/
  surf2Dwrite ( out_color, colortexture, coords.x*sizeof(float4), coords.y );
  surf2Dwrite ( out_depth, depthtexture, coords.x*sizeof(float), coords.y );
}
