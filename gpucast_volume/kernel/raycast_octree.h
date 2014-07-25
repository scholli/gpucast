/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_octree.h
*  project    : gpucast
*  description:
*
********************************************************************************/

#include <math/mult.h>
#include <math/length.h>
#include <math/operator.h>

#include <octree/bbox_entry_from_exit.h>
#include <octree/bbox_exit_from_entry.h>
#include <octree/octree_lookup.h>
#include <octree/ocnode_size.h>
#include <octree/raycast_ocnode.h>

#include <gpucast/volume/isosurface/renderconfig.hpp>

// just for numerical robustness -> abort after max. iterations
// practically NEVER happens!
unsigned const MAX_OCNODE_INTERSECTIONS = 1024;

///////////////////////////////////////////////////////////////////////////////
__device__
void raycast_octree ( gpucast::renderconfig const&  config,
                      gpucast::bufferinfo const&    info,
                      int2                          coords,
                      float4 const*                 matrixbuffer,
                      uint4 const*                  nodebuffer,
                      uint4 const*                  facelistbuffer,
                      float4 const*                 bboxbuffer,
                      float const*                  limitbuffer,
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
    /*if ( external_depth < 1.0f ) 
    {
      surf2Dwrite ( external_color, colortexture, coords.x*sizeof(float4), coords.y );
      surf2Dwrite ( external_depth, depthtexture, coords.x*sizeof(float), coords.y );
    }*/
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
  float4    ray_origin                 = mult_mat4_float4 ( modelviewinverse, float4_t(0.0f, 0.0f, 0.0f, 1.0f) );
  float4    ray_direction              = float4_t ( input_position.x - ray_origin.x, input_position.y - ray_origin.y, input_position.z - ray_origin.z, 0.0f ); 
  ray_direction                        = normalize(ray_direction);

  float3    octree_min                 = float3_t ( config.bbox_min[0], config.bbox_min[1], config.bbox_min[2] );
  float3    octree_max                 = float3_t ( config.bbox_max[0], config.bbox_max[1], config.bbox_max[2] );
  float3    octree_size                = octree_max - octree_min;
  unsigned const max_octree_depth      = config.max_octree_depth;
  float const min_octree_coord         = min ( octree_size.x, min ( octree_size.z, octree_size.y ) );
  float const epsilon_offset           = min_octree_coord / pow(2.0f, float(max_octree_depth+1));

  float4 rayexit_octree_coordinates  = float4_t( ( input_position.x - octree_min.x ) / octree_size.x,
                                                 ( input_position.y - octree_min.y ) / octree_size.y,
                                                 ( input_position.z - octree_min.z ) / octree_size.z,
                                                   1.0f );    
  float4 ray_entry_old = ray_origin;
  float4 ray_entry;
  float4 ray_entry_normal;
  bbox_entry_from_exit ( input_position, ray_direction, octree_min, octree_max, ray_entry, ray_entry_normal );

  float4 rayentry_octree_coordinates = float4_t( ( ray_entry.x      - octree_min.x ) / octree_size.x,
                                                 ( ray_entry.y      - octree_min.y ) / octree_size.y,
                                                 ( ray_entry.z      - octree_min.z ) / octree_size.z,
                                                   1.0f );
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

  /*************************************
  * raycast octree representation
  **************************************/
  while ( continue_raytracing )
  {
    // move node entry point a bit along normal to do clean octree lookup
    float3 entry_with_offset = float3_t ( ray_entry.x - epsilon_offset * ray_entry_normal.x,
                                          ray_entry.y - epsilon_offset * ray_entry_normal.y,
                                          ray_entry.z - epsilon_offset * ray_entry_normal.z );

    /*************************************
    * get ocnode for current ray position
    **************************************/
    int     octree_depth = 0;
    float3  ocnode_min;
    float3  ocnode_max;
    uint4 node  = octree_lookup ( nodebuffer, limitbuffer, 1, entry_with_offset, octree_min, octree_max, config.isovalue, octree_depth, ocnode_min, ocnode_max );

    /*************************************
    * compute ocnode exit 
    **************************************/
    float4  ray_exit;
    float4  ray_exit_normal;  
    float   ray_exit_t = 0.0f;
    bbox_exit_from_entry ( ray_entry, ray_direction, ocnode_min, ocnode_max, ray_exit, ray_exit_normal, ray_exit_t );

    bool is_inner_node = (node.x <= 1);

    /*************************************
    * if inner node -> no leafs to examine -> skip node
    **************************************/
    if ( is_inner_node ) // if inner node hit -> intersect inner node and continue at exit point
    {
      // set entry to exit point
      //out_color    = out_color + out_opacity * config.boundary_opacity * float4_t ( 0.0, 0.0, 1.0f, 1.0);
      //out_opacity *= ( 1.0f - config.boundary_opacity );
    } else { // if leaf node hit

      /*************************************
      * if contains outer surface or face with according attribute bounds -> examine node
      **************************************/
      float attrib_min = limitbuffer[node.y  ];
      float attrib_max = limitbuffer[node.y+1];

      bool contains_isovalue   = (config.isovalue > attrib_min) && (config.isovalue < attrib_max);
      bool contains_outer_face = (node.x == 3);
      if ( contains_isovalue || contains_outer_face ) // if node does contain potential faces
      {
        // iterate face list
        raycast_ocnode ( node,
                         current_state,
                         facelistbuffer, 
                         bboxbuffer, 
                         limitbuffer,
                         surfacedatabuffer,
                         surfacepointsbuffer,
                         volumedatabuffer,
                         volumepointsbuffer,
                         attributedatabuffer,
                         attributepointsbuffer,
                         ray_entry,
                         ray_exit,
                         ray_exit_t,
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
                         octree_depth,
                         ocnode_min,
                         ocnode_max );

      } else { 
        // if node does NOT contain potential faces
      }
    }

    /*************************************
    * reached external opaque surface -> end ray casting 
    * external color has been applied in raycast_ocnode()
    **************************************/
    if ( out_depth > external_depth )
    {
      continue_raytracing = false;
    }

    /*************************************
    * go into next node -> check for octree exit 
    *   -> if exit outside octree -> stop ray casting
    **************************************/
    float3 exit_with_offset = float3_t ( ray_exit.x + epsilon_offset * ray_exit_normal.x, 
                                         ray_exit.y + epsilon_offset * ray_exit_normal.y, 
                                         ray_exit.z + epsilon_offset * ray_exit_normal.z ); 
    if ( exit_with_offset.x > octree_max.x || exit_with_offset.x < octree_min.x ||
         exit_with_offset.y > octree_max.y || exit_with_offset.y < octree_min.y ||
         exit_with_offset.z > octree_max.z || exit_with_offset.z < octree_min.z )
    {
      //if ( out_color.w == 0.0f ) {
      //  out_color    = float4_t ( 1.0, 1.0, 1.0, 1.0);
      //}
      continue_raytracing = false;
    }
    
    if ( out_opacity < out_saturation_reached )
    {
      continue_raytracing = false;
    }

    /*************************************
    * critical exit :  total number of iterations exceeds maximum -> abort
    **************************************/
    if ( MAX_OCNODE_INTERSECTIONS < iterations++ )
    {
      out_color           = float4_t ( 1.0, 0.0, 0.0, 1.0);
      continue_raytracing = false;
    }

    ray_entry_old    = ray_entry;
    ray_entry        = ray_exit;

    float t_old = ( ray_entry_old.z - ray_origin.z ) / ray_direction.z;
    float t_new = ( ray_entry.z - ray_origin.z )     / ray_direction.z;

    if ( t_old >= t_new ) // no progress
    {
      out_color           = float4_t ( 1.0, 0.0, 0.0, 1.0);
      continue_raytracing = false;
    }

    ray_entry_normal = -1.0f * ray_exit_normal;
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
