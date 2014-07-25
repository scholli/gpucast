/********************************************************************************
*
* Copyright (C) 2009-2012 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : raycast_isosurface.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_RAYCAST_ISOSURFACE_H
#define LIBGPUCAST_CUDA_RAYCAST_ISOSURFACE_H

#include <math/horner_surface.h>
#include <math/cross.h>
#include <math/normalize.h>
#include <math/compute_depth.h>
#include <math/mult.h>
#include <math/operator.h>
#include <math/horner_volume.h>
#include <math/transferfunction.h>

#include <isosurface/target_function.h>
#include <isosurface/search_volume_for_iso_surface.h>

#include <shade/faceforward.h>

#include <octree/bbox_intersection.h>

#include <fragmentlist/load_volume_data.h>
#include <fragmentlist/classify_transition.h>

#include <gpucast/volume/isosurface/renderconfig.hpp>

__device__ inline
void raycast_isosurface ( gpucast::renderconfig  config,
                          gpucast::bufferinfo    info,
                          float4 const*          modelview,
                          float4 const*          normalmatrix,
                          uint4 const*           surfacedatabuffer,
                          float4 const*          surfacepointsbuffer,
                          float4 const*          volumedatabuffer,
                          float4 const*          volumepointsbuffer,
                          float4 const*          attributedatabuffer,
                          float2 const*          attributepointsbuffer,
                          bbox_intersection*     intersections, 
                          unsigned               intersections_found,
                          float4 const&          external_color,
                          float const&           external_depth,
                          float&                 out_depth,
                          float4&                out_color,
                          float&                 out_opacity,
                          raystate&              ray_state,
                          float4 const&          ray_entry,
                          float4 const&          ray_exit,
                          float                  ray_exit_t,
                          float4 const&          ray_direction )
{
  out_opacity *= 0.999f;
  out_depth = 0.3f;

  for ( unsigned i = 0; i != intersections_found; ++i )
  {
    // gather information about hit
    unsigned  surface_index = intersections[i].surface_index;
    float2    uv            = intersections[i].uv;



    unsigned  surfacemesh_id= surfacedatabuffer[surface_index+1].y;
    //unsigned  volume_id     = surfacedatabuffer[surface_index].z;

    uint2     order         = uint2_t ( surfacedatabuffer[surface_index+2].y, 
                                        surfacedatabuffer[surface_index+2].z );

    // compute hit normal and partial derivatives etc.
    float4 face_point, face_du, face_dv;

    horner_surface_derivatives ( surfacepointsbuffer, surfacemesh_id, order, uv, face_point, face_du, face_dv); 
    float3 face_normal = normalize ( cross ( float3_t(face_du.x, face_du.y, face_du.z), 
                                             float3_t(face_dv.x, face_dv.y, face_dv.z) ) );

    float4 face_point_world = mult_mat4_float4 ( modelview, face_point );
    float face_depth = compute_depth_from_world_coordinates ( face_point_world, config.nearplane, config.farplane );

    if ( face_depth > external_depth )
    {
      out_color = out_color + out_opacity * external_color;
      out_depth = external_depth;
      out_opacity = 0.0f;
      break;
    }

    bool same_surface = surface_index == ray_state.surface;
    bool same_depth   = fabs ( ray_state.depth - face_depth ) < config.newton_epsilon;
    bool is_identical = same_surface && same_depth;

 
    if ( is_identical )
    {
      // do nothing
    } else {
      /********************************************************************************
      * retrieve information about fragment, volume and surface
      ********************************************************************************/         
      unsigned surface_mesh_id;
      unsigned surface_type;
      uint2    surface_order;
      bool     surface_is_outer = false;
      unsigned object_id;

      unsigned volume_data_id;
      unsigned volume_points_id;
      //unsigned volume_unique_id;
      uint3    volume_order;
      float    volume_bbox_size;

      unsigned attribute_data_id;         
      unsigned attribute_points_id;         
      unsigned adjacent_volume_data_id;
      unsigned adjacent_attribute_data_id;
      float    adjacent_volume_bbox_size;

      load_volume_data ( surfacedatabuffer, volumedatabuffer, attributedatabuffer, surface_index, 
                         &surface_order, &surface_type, &surface_mesh_id, &surface_is_outer,
                         &volume_data_id, &volume_points_id, &attribute_data_id, &attribute_points_id, &volume_order, &volume_bbox_size, 
                         &adjacent_volume_data_id, &adjacent_attribute_data_id, &adjacent_volume_bbox_size, &object_id );

      /********************************************************************************
      * map 2D face parameter to 3d volume parameter space 
      ********************************************************************************/         
      uint4  surface_param_info           = surfacedatabuffer [surface_index + 3];
      uint3   uvw_swizzle                 = uint3_t ( surface_param_info.x, surface_param_info.y, surface_param_info.z);
      float   w                           = clamp ( float(surface_param_info.w), 0.0f, 1.0f );
      float   tmp_uvw[3];
      tmp_uvw[uvw_swizzle.x]              = uv.x;
      tmp_uvw[uvw_swizzle.y]              = uv.y;
      tmp_uvw[uvw_swizzle.z]              = w;
      float3 uvw                          = float3_t( tmp_uvw[0], tmp_uvw[1], tmp_uvw[2] ); 
      uvw                                 = clamp ( uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f) );
    
      /********************************************************************************
      * shade outer surface
      ********************************************************************************/    
      if ( surface_is_outer ) 
      {
        float4 lightpos    = float4_t(0.0f, 0.0f, 0.0f, 1.0f); 
        float3 L           = normalize ( float3_t(lightpos.x, lightpos.y, lightpos.z) - float3_t(face_point_world.x, face_point_world.y, face_point_world.z) );

        float4 surface_normal = mult_mat4_float4 ( normalmatrix, float4_t ( face_normal.x, face_normal.y, face_normal.z, 0.0f) );
        float3 N              = normalize ( float3_t(surface_normal.x, surface_normal.y, surface_normal.z) );
        N                     = faceforward ( float3_t(-face_point_world.x, -face_point_world.y, -face_point_world.z), N );
        
        float diffuse      = dot (N , L);
        diffuse            = ( diffuse * 0.5f ) + 0.5f;
        float2 attrib      = horner_volume<float2,1> ( attributepointsbuffer, attribute_points_id, volume_order, uvw );
        float4 attrib_color = float4_t(attrib.x, 0.0, 0.0, 1.0);
        
        if ( config.show_isosides ) 
        {
          if ( target_function ( attrib.x ) < target_function ( config.isovalue ) ) 
          {
            attrib_color = float4_t ( 1.0f, 0.0f, 0.0f, 1.0f);
          } else { 
            attrib_color = float4_t ( 0.0f, 1.0f, 0.0f, 1.0f);
          }
        } else {
          float norm_attrib  = normalize ( attrib.x, config.attrib_min, config.attrib_max );
          attrib_color       = transferfunction ( norm_attrib );
        }

        out_depth          = compute_depth_from_world_coordinates ( face_point_world, config.nearplane, config.farplane );
        out_color          = out_color + config.boundary_opacity * out_opacity * diffuse * attrib_color;
        out_opacity        = out_opacity * (1.0f - config.boundary_opacity);
      }

      /********************************************************************************
      * check for search interval
      ********************************************************************************/
      float   attribute_min               = attributedatabuffer[attribute_data_id].x;
      float   attribute_max               = attributedatabuffer[attribute_data_id].y;
      bool    contains_iso_value          = is_in_range ( config.isovalue, attribute_min, attribute_max );
      bool    adjacent_contains_iso_value = false;

      if ( adjacent_volume_data_id ) // has neighbor volume
      {
        float  adjacent_attribute_min    = attributedatabuffer[adjacent_attribute_data_id].x;
        float  adjacent_attribute_max    = attributedatabuffer[adjacent_attribute_data_id].y;
        adjacent_contains_iso_value       = is_in_range ( config.isovalue, adjacent_attribute_min, adjacent_attribute_max );
      }

      /********************************************************************************
      * determine interval
      ********************************************************************************/
      bool volume_to_volume                   = ray_state.volume          == volume_data_id;
      bool volume_to_adjacent_volume          = ray_state.volume          == adjacent_volume_data_id;
      bool adjacent_volume_to_adjacent_volume = ray_state.adjacent_volume == adjacent_volume_data_id;
      bool adjacent_volume_to_volume          = ray_state.adjacent_volume == volume_data_id;

      // set search parameter to default
      unsigned search_volume_data_id      = 0;
      unsigned search_volume_points_id    = 0;
      unsigned search_attribute_data_id   = 0;
      unsigned search_attribute_points_id = 0;
      uint3    search_order               = uint3_t(0, 0, 0);
      float3   search_uvw                 = float3_t(0.0f, 0.0f, 0.0f);
      float4   search_start               = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
      float4   search_stop                = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
      float    search_bbox_size           = 0.0f;

      bool     search_interval_found = false;

      // CASE 1 :
      if ( volume_to_volume && contains_iso_value )
      {
        //out_color        = out_color + config.boundary_opacity * out_opacity * float4_t(1.0, 0.0, 0.0, 1.0);
        //out_opacity      = out_opacity *  (1.0f - config.boundary_opacity);

        search_interval_found      = true;
        search_volume_data_id      = volume_data_id;
        search_volume_points_id    = volume_points_id;
        search_attribute_data_id   = attribute_data_id;
        search_attribute_points_id = attribute_points_id;
        search_order               = volume_order;
        search_uvw                 = ray_state.uvw;
        search_start               = ray_state.point;
        search_stop                = face_point;
        search_bbox_size           = volume_bbox_size;
      }

      // CASE 2 :
      if ( volume_to_adjacent_volume && adjacent_contains_iso_value )
      {
        //out_color        = out_color + config.boundary_opacity * out_opacity * float4_t(1.0, 0.0, 0.0, 1.0);
        //out_opacity      = out_opacity *  (1.0f - config.boundary_opacity);

        search_interval_found      = true;
        search_volume_data_id      = adjacent_volume_data_id;
        search_volume_points_id    = floatBitsToInt(volumedatabuffer[adjacent_volume_data_id].x);
        search_attribute_data_id   = adjacent_attribute_data_id;
        search_attribute_points_id = floatBitsToInt(attributedatabuffer[adjacent_attribute_data_id].z);
        search_order               = volume_order;
        search_uvw                 = ray_state.uvw;
        search_start               = ray_state.point;
        search_stop                = face_point;
        search_bbox_size           = adjacent_volume_bbox_size;
      }

      // CASE 3 :
      if ( adjacent_volume_to_adjacent_volume && adjacent_contains_iso_value )
      {
        //out_color        = out_color + config.isosurface_opacity * out_opacity * float4_t(0.0, 1.0, 0.0, 1.0);
        //out_opacity      = out_opacity *  (1.0f - config.isosurface_opacity);

        search_interval_found      = true;
        search_volume_data_id      = adjacent_volume_data_id;
        search_volume_points_id    = floatBitsToInt(volumedatabuffer[adjacent_volume_data_id].x);
        search_attribute_data_id   = adjacent_attribute_data_id;
        search_attribute_points_id = floatBitsToInt(attributedatabuffer[adjacent_attribute_data_id].z);
        search_order               = volume_order;
        search_uvw                 = ray_state.uvw;

        //if ( surface_type == 1 ) search_uvw.x = 0.0f; // coming from volume at umax side -> umin
        //if ( surface_type == 3 ) search_uvw.y = 0.0f; // coming from volume at vmax side -> vmin
        //if ( surface_type == 5 ) search_uvw.z = 0.0f; // coming from volume at wmax side -> wmin

        search_start          = ray_state.point;
        search_stop           = face_point;
        search_bbox_size      = adjacent_volume_bbox_size;
      }
      

      // CASE 4 :
      if ( adjacent_volume_to_volume && contains_iso_value )
      {
        //out_color        = out_color + config.isosurface_opacity * out_opacity * float4_t(0.0, 0.0, 1.0, 1.0);
        //out_opacity      = out_opacity *  (1.0f - config.isosurface_opacity);

        search_interval_found      = true;
        search_volume_data_id      = volume_data_id;
        search_volume_points_id    = volume_points_id;
        search_attribute_data_id   = attribute_data_id;
        search_attribute_points_id = attribute_points_id;
        search_order               = volume_order;
        search_uvw                 = ray_state.uvw;

        //if ( surface_type == 1 ) search_uvw.x = 0.0f; // coming from volume at umax side -> umin
        //if ( surface_type == 3 ) search_uvw.y = 0.0f; // coming from volume at vmax side -> vmin
        //if ( surface_type == 5 ) search_uvw.z = 0.0f; // coming from volume at wmax side -> wmin

        search_start          = ray_state.point;
        search_stop           = face_point;
        search_bbox_size      = volume_bbox_size;
      }

      /********************************************************************************
      * raytrace isosurface 
      ********************************************************************************/
      float4 iso_hit_position;
      float2 iso_hit_attribute;
      float4 iso_hit_normal;
      float3 iso_hit_uvw;

      unsigned total_samples = 0;

      if ( search_interval_found )
      {
        bool continue_isosurface_search                   = true;
        int isosurface_intersections_per_volume           = 0;
        int const max_isosurface_intersections_per_volume = 4;

        while ( continue_isosurface_search && isosurface_intersections_per_volume < max_isosurface_intersections_per_volume )
        {
          ++isosurface_intersections_per_volume;
#if 1
          bool assert_conditions = search_volume_points_id    < info.volumepoints_size - 9 &&
                                   search_attribute_points_id < info.attributepoints_size - 9 &&
                                   search_order.x < 5 && search_order.y < 5 && search_order.z < 5;
          if ( assert_conditions ) {
          
            continue_isosurface_search = search_volume_for_iso_surface ( volumepointsbuffer,
                                                                         attributepointsbuffer,
                                                                         search_volume_points_id,
                                                                         search_attribute_points_id,
                                                                         search_order,
                                                                         search_uvw,
                                                                         config.isovalue,
                                                                         search_start,
                                                                         search_stop,
                                                                         config.adaptive_sampling,
                                                                         search_bbox_size,
                                                                         config.steplength_min,
                                                                         config.steplength_max,
                                                                         config.steplength_scale,
                                                                         config.screenspace_newton_error,
                                                                         config.newton_epsilon,
                                                                         config.newton_iterations,
                                                                         config.max_binary_steps,
                                                                         &iso_hit_position,
                                                                         &iso_hit_attribute,
                                                                         &iso_hit_normal,
                                                                         &iso_hit_uvw,
                                                                         &search_start,
                                                                         &search_uvw,
                                                                         total_samples );
          } else {
            continue_isosurface_search = false;
          }
                                     
#else
          continue_isosurface_search = false;
#endif         
          if ( continue_isosurface_search )
          {
            // shade
            float4 lightpos    = float4_t(0.0f, 0.0f, 0.0f, 1.0f); // light from camera
            float4 pworld      = mult_mat4_float4 ( modelview, iso_hit_position );
            iso_hit_normal     = mult_mat4_float4 ( normalmatrix, iso_hit_normal );
                            
            //current_depth      = compute_depth_from_world_coordinates ( pworld, config.nearplane, config.farplane );
            float3 L           = normalize ( float3_t ( lightpos.x - pworld.x, lightpos.y - pworld.y, lightpos.z - pworld.z ));
            float3 N           = normalize ( float3_t ( iso_hit_normal.x, iso_hit_normal.y, iso_hit_normal.z ));

            N = dot ( float3_t ( -pworld.x, -pworld.y, -pworld.z), N ) < 0.0f ? -N : N;
            float diffuse      = dot (N , L);

            float relative_attrib     = normalize ( iso_hit_attribute.x, config.attrib_min, config.attrib_max );
            float4 relative_iso_color = transferfunction ( relative_attrib );
            relative_iso_color.w = 1.0f / diffuse;

            out_depth          = compute_depth_from_world_coordinates ( pworld, config.nearplane, config.farplane );
            out_color          = out_color + config.isosurface_opacity * out_opacity * float4_t(diffuse * relative_iso_color.x, diffuse * relative_iso_color.y, diffuse * relative_iso_color.z, 1.0);
            out_opacity        = out_opacity *  (1.0f - config.isosurface_opacity);
          } 
        } // while there still might be an isosurface transition in interval
      } // found_exit fragment -> search interval for iso surface


      /********************************************************************************
      * apply current state to ray state
      ********************************************************************************/ 
      ray_state.surface          = surface_index;
      ray_state.volume           = volume_data_id;
      ray_state.adjacent_volume  = adjacent_volume_data_id;
      ray_state.depth            = face_depth;
      ray_state.point            = face_point;
      ray_state.uvw              = uvw;

     
    } // found new intersection

  }

}

#endif // LIBGPUCAST_CUDA_RAYCAST_ISOSURFACE_H

