/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : raycast_volume_tree.glsl.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable
#extension GL_EXT_bindable_uniform : enable
#extension GL_NV_shader_buffer_load : enable


/********************************************************************************
* constants
********************************************************************************/
#define OCTREE_OFFSET_EPSILON               0.01 // should be 1/2^(depth_of_tree + 1)
#define OBB_INTERSECT_AVOID_ZFIGHT_EPSILON  0.0001
#define MAX_OCTREE_TRAVERSAL                64
#define MAX_OBB_INTERSECTIONS_PER_OCNODE    64
#define INFINITY                            1.0e32
#define FLOAT_EPSILON                       0.00001                             

/********************************************************************************
* uniforms
********************************************************************************/
// volume information buffers
uniform isamplerBuffer  treebuffer;
uniform isamplerBuffer  volumelistbuffer;

uniform samplerBuffer   volumebuffer;
uniform samplerBuffer   surfacebuffer;
uniform samplerBuffer   boundingboxbuffer;
uniform samplerBuffer   databuffer;
uniform samplerBuffer   limitbuffer;

uniform sampler2D       transfertexture;

layout (size4x32) uniform image2D         colorattachment0;
layout (size1x32) uniform image2D         depthattachment;

// model information
uniform vec4            octree_bounds_min;
uniform vec4            octree_bounds_max;
uniform int             octree_depth;

// tranformation matrices
uniform mat4            modelviewprojectionmatrix;
uniform mat4            modelviewmatrix;
uniform mat4            modelviewmatrixinverse;
uniform mat4            normalmatrix;

// parametrization
uniform vec4            iso_value;
uniform float           fixed_newton_epsilon;
uniform int             newton_max_iterations;

uniform bool            adaptive_sampling;
uniform bool            adaptive_newton_epsilon;
uniform float           min_sample_distance;
uniform float           max_sample_distance;
uniform float           adaptive_sample_scale;

uniform int             max_steps_binary_search;
uniform vec4            attribute_min;
uniform vec4            attribute_max;
uniform int             component;

// gl state
uniform float           nearplane;
uniform float           farplane;
uniform int             width;
uniform int             height;
uniform int             gradient_distance;
uniform bool            compute_gradients;


/********************************************************************************
* input
********************************************************************************/
in vec4 fragcolor;
in vec4 fragposition;
in vec4 fragobjectcoordinates;
in vec3 fragnormal;
in vec4 fragtexcoords;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 out_color;

/********************************************************************************
* functions
********************************************************************************/
#include "./libgpucast/glsl/base/compute_depth.frag"
#include "./libgpucast/glsl/base/faceforward.frag"

#include "./libgpucast/glsl/math/raygeneration.glsl.frag"
#include "./libgpucast/glsl/math/adjoint.glsl.frag"
#include "./libgpucast/glsl/math/euclidian_space.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./libgpucast/glsl/math/horner_volume.glsl.frag"
#include "./libgpucast/glsl/math/newton_surface.glsl.frag"
#include "./libgpucast/glsl/math/newton_volume.glsl.frag"

#include "./libgpucast/glsl/isosurface/target_function.frag"
#include "./libgpucast/glsl/isosurface/compute_iso_normal.frag"
#include "./libgpucast/glsl/isosurface/compute_next_sampling_position.frag"
#include "./libgpucast/glsl/isosurface/trilinear_interpolation.frag"
#include "./libgpucast/glsl/isosurface/clip_ray_at_nearplane.frag"
#include "./libgpucast/glsl/isosurface/compute_entry_from_exit.frag"
#include "./libgpucast/glsl/isosurface/compute_exit_from_entry.frag"
#include "./libgpucast/glsl/isosurface/intersect_obb.frag"
#include "./libgpucast/glsl/isosurface/octree_compute_node.frag"
#include "./libgpucast/glsl/isosurface/octree_lookup.frag"
#include "./libgpucast/glsl/isosurface/find_nearest_obb_intersection.frag"
#include "./libgpucast/glsl/isosurface/binary_search_for_isosurface.frag"
#include "./libgpucast/glsl/isosurface/newton_search_for_isosurface.frag"
#include "./libgpucast/glsl/isosurface/secant_search_for_isosurface.frag"
#include "./libgpucast/glsl/isosurface/validate_isosurface_intersection.frag"
#include "./libgpucast/glsl/isosurface/node_potentially_contains_isosurface.frag"


/********************************************************************************
* shader for raycasting a beziervolume
********************************************************************************/
void main(void)
{
  /////////////////////////////////////////////////////////////////////////////
  // front face culling
  /////////////////////////////////////////////////////////////////////////////
  if ( dot(fragposition.xyz, fragnormal.xyz) < 0.0) 
  {
    discard;
  }

  /////////////////////////////////////////////////////////////////////////////
  // ray setup
  /////////////////////////////////////////////////////////////////////////////
  vec4 octree_scale   = octree_bounds_max - octree_bounds_min;
  vec3 bboxmin        = octree_bounds_min.xyz;
  vec3 bboxmax        = octree_bounds_max.xyz;
  
  // ray setup in object coordinates
  vec4 ray_exit      = fragobjectcoordinates; // get exit through rendering backfaces
  vec4 ray_origin    = modelviewmatrixinverse * vec4(0.0, 0.0, 0.0, 1.0);    
  vec4 ray_direction = vec4(normalize(ray_exit.xyz - ray_origin.xyz), 0.0);

  // precompute scalar for adaptive newton error
  vec4 ray_exit_halfpixel_offset      = fragobjectcoordinates + 0.5 * min( dFdx(fragobjectcoordinates), dFdy(fragobjectcoordinates) );
  vec4 raydirection_halfpixel_offset  = vec4(normalize(ray_exit_halfpixel_offset.xyz - ray_origin.xyz), 0.0);
  float angle_halfpixel               = dot ( raydirection_halfpixel_offset.xyz, ray_direction.xyz );

  float t            = 0.0;
  vec3 entry_normal  = vec3(0.0);

  // compute entry into octree
  vec4 ray_entry     = compute_entry_from_exit( ray_exit, 
                                                ray_direction, 
                                                bboxmin, 
                                                bboxmax,
                                                t,
                                                entry_normal);

  // clip agains near plane
  ray_entry = clip_ray_at_nearplane ( modelviewmatrix, modelviewmatrixinverse, ray_entry, ray_direction, nearplane);

  // compute planes for newton iteration
  vec3    n1 = vec3(0.0);
  vec3    n2 = vec3(0.0);
  float   d1 = 0.0;
  float   d2 = 0.0;
  raygen (ray_origin, ray_direction, n1, n2, d1, d2);

  /////////////////////////////////////////////////////////////////////////////
  // octree traversal
  /////////////////////////////////////////////////////////////////////////////
  ivec4 node                    = ivec4(0);
  bool  iso_found               = false;
  bool  continue_ray_traversal  = true;

  int   iters                   = 0;
  vec4  iso_hit_data            = vec4(0.0);
  vec4  iso_hit_position        = vec4(0.0);
  vec4  iso_hit_normal          = vec4(0.0);

  // traverse octree to find appropriate ocnode
  while ( continue_ray_traversal ) 
  {
    // to maintain numerical stability while searching for ocnode -> translate ray origin perpendicular to last hit
    vec3 octree_lookup_offset = entry_normal.xyz * octree_scale.xyz * OCTREE_OFFSET_EPSILON;

    // store depth in which a node was found
    int ocdepth = 0;

    // traverse octree and find current ocnode for current origin of the ray
    node                      = octree_lookup           ( treebuffer, 
                                                          ray_entry.xyz + octree_lookup_offset, 
                                                          octree_bounds_min.xyz, 
                                                          octree_bounds_max.xyz,  
                                                          ocdepth );

    // compute size and position of current ocnode's bounding box in object space
    compute_ocnode  ( ray_entry.xyz + octree_lookup_offset, 
                      octree_bounds_min.xyz, 
                      octree_bounds_max.xyz, 
                      ocdepth, 
                      bboxmin, 
                      bboxmax);

    // compute exit of ray where the search should continue if no iso value is found in this node
    vec3 exit_normal          = vec3(0.0);
    float t_max               = 0.0;

    // compute exit out of ocnode and max t parameter within node
    vec4 ocnode_exit             = compute_exit_from_entry (  ray_entry, 
                                                              ray_direction, 
                                                              bboxmin, 
                                                              bboxmax, 
                                                              t_max,
                                                              exit_normal );

    // success -> found non-empty node -> try to find isosurface in this node
    bool node_contains_data = node.z > 0;

    if ( node_contains_data ) 
    {
      // compare iso value with iso limits of all volumes overlapping this node
      bool node_potentially_contains_iso_value = node_potentially_contains_isosurface ( limitbuffer,
                                                                                        node, 
                                                                                        ray_entry, 
                                                                                        ray_direction, 
                                                                                        octree_bounds_min, 
                                                                                        octree_bounds_max, 
                                                                                        iso_value );

      // if it includes volumes with iso value -> intersect obb's of volume
      if ( node_potentially_contains_iso_value ) 
      {
        vec4  obb_intersect_in              = vec4(0.0);
        vec4  obb_intersect_out             = vec4(0.0);
        vec4  obb_size                      = vec4(0.0);
        bool  found_obb_intersection        = true;
        int   inner_ocnode_iterations       = 0;
        ivec4 last_volume_list1             = ivec4(-1);
        ivec4 last_volume_list2             = ivec4(-1);

        // try to find an iso surface intersection within bounding box
        while ( found_obb_intersection && 
                inner_ocnode_iterations < MAX_OBB_INTERSECTIONS_PER_OCNODE ) 
        {
          // parameter values from obb intersection as starting and end points for newton iteration
          vec3 obb_uvw_in           = vec3(0.0);
          vec3 obb_uvw_out          = vec3(0.0);
          float obb_t_in            = 0.0;
          float obb_t_out           = 0.0;
          ivec4 volume              = ivec4(0);

          found_obb_intersection    = find_nearest_obb_intersection_including_iso_value ( volumelistbuffer,
                                                                                          limitbuffer,
                                                                                          node, 
                                                                                          ray_entry, 
                                                                                          ray_direction, 
                                                                                          t_max, 
                                                                                          iso_value, 
                                                                                          obb_t_in,
                                                                                          obb_uvw_in, 
                                                                                          obb_intersect_in,
                                                                                          obb_t_out,
                                                                                          obb_uvw_out,
                                                                                          obb_intersect_out,
                                                                                          obb_size,
                                                                                          last_volume_list1,
                                                                                          last_volume_list2,
                                                                                          volume );

          // successful intersection with next oriented bounding box
          if ( found_obb_intersection ) 
          {
            // iterate through volume and try to find iso surface
            bool  found_iso_surface     = false;
            vec4  sample_start          = ray_entry + max( 0.0, obb_t_in) * ray_direction;
            vec4  sample_end            = ray_entry + min( t_max, obb_t_out) * ray_direction;

            float sampled_width         = length(sample_end.xyz - sample_start.xyz);
            float obb_diameter          = length(obb_size.xyz);
            bool  sample_end_reached    = false;

            int   max_samples           = int(ceil(sampled_width / (min_sample_distance * obb_diameter) ) ) + 1;
            bool  last_sample_valid     = false;
            bool  first_sample_valid    = false;
            vec4  last_sample_position  = vec4(0.0);
            vec4  last_sample_data      = vec4(0.0);

            vec3 uvw_next  = vec3(0.0);

            // if ray entry is in obb -> interpolate
            if ( obb_t_in < 0.0 )
            {
              float a     = abs(obb_t_in) / (obb_t_out - obb_t_in);
              obb_uvw_in  = mix(obb_uvw_in, obb_uvw_out, a);
            }
            vec3 uvw_start = obb_uvw_in;

            vec4 sample_data = vec4(0.0);
            vec4 point       = vec4(0.0);
            vec4 ddu         = vec4(0.0);
            vec4 ddv         = vec4(0.0);
            vec4 ddw         = vec4(0.0);
            vec4 du          = vec4(0.0);
            vec4 dv          = vec4(0.0);
            vec4 dw          = vec4(0.0);

            for (int sample_index = 0; sample_index < max_samples; ++sample_index)
            {
              vec4 sample_position = vec4(0.0);
             
              if ( !first_sample_valid )
              {
                sample_position     = sample_start;
                first_sample_valid  = true;
              } else {
                if ( adaptive_sampling )
                {
                  sample_position = compute_sampling_position_adaptively (last_sample_position,
                                                                          ray_direction,
                                                                          obb_diameter,
                                                                          min_sample_distance,
                                                                          max_sample_distance,
                                                                          adaptive_sample_scale,
                                                                          iso_value,
                                                                          last_sample_data,
                                                                          du, dv, dw,
                                                                          ddu, ddv, ddw );
                } else {
                  sample_position = compute_sampling_position ( last_sample_position,
                                                                ray_direction,
                                                                min_sample_distance,
                                                                obb_diameter );
                }
              }
           
              // if sampling end was reached last iteration break now
              if (sample_end_reached) {
                break;
              }
              sample_end_reached = length(sample_position.xyz - ray_entry.xyz) > obb_t_out;

              float newton_epsilon = fixed_newton_epsilon;
              //float newton_epsilon = length(sample_position.xyz - ray_origin.xyz) * sqrt(1.0 - angle_halfpixel*angle_halfpixel);

              //bool newton_success = false;
              bool newton_success = newton_volume ( volumebuffer, 
                                                    volume.x,
                                                    volume.y,
                                                    volume.z,
                                                    volume.w,
                                                    uvw_start,
                                                    uvw_next,
                                                    sample_position,
                                                    point,
                                                    du,
                                                    dv,
                                                    dw,
                                                    n1,
                                                    n2,
                                                    ray_direction.xyz,
                                                    d1,
                                                    d2,
                                                    newton_epsilon,
                                                    newton_max_iterations );
                            
              evaluateVolume ( databuffer, volume.x, volume.y, volume.z, volume.w, uvw_next.x, uvw_next.y, uvw_next.z, sample_data, ddu, ddv, ddw );

              if (  ( target_function ( last_sample_data ) - target_function ( iso_value ) ) * 
                    ( target_function ( sample_data )      - target_function ( iso_value ) ) < 0.0
                    && last_sample_valid        /* valid first sample */
                    )
              {
                vec4 iso_position = vec4(0.0);
                vec4 iso_data     = vec4(0.0);
                vec3 iso_uvw      = vec3(0.0);
                vec4 iso_normal   = vec4(0.0);

                binary_search_for_isosurface        ( 
                //newton_search_for_isosurface        (
                //secant_search_for_isosurface        (
                                                      volumebuffer, 
                                                      databuffer, 
                                                      volume.x,
                                                      volume.x,
                                                      volume.yzw,
                                                      ray_entry,
                                                      ray_direction,
                                                      uvw_start, 
                                                      uvw_next, 
                                                      sample_position, 
                                                      last_sample_position, 
                                                      sample_data, 
                                                      last_sample_data, 
                                                      iso_value,
                                                      newton_epsilon,
                                                      newton_max_iterations,
                                                      max_steps_binary_search,
                                                      d1,
                                                      d2,
                                                      n1,
                                                      n2,
                                                      iso_uvw, 
                                                      iso_position, 
                                                      iso_data,
                                                      iso_normal );

                found_iso_surface = validate_isosurface_intersection (volumebuffer, 
                                                                      databuffer, 
                                                                      volume, 
                                                                      ray_entry,
                                                                      ray_direction, 
                                                                      iso_uvw,
                                                                      iso_value,
                                                                      0.0005f);

                if ( found_iso_surface ) 
                {
                  iso_hit_data                = iso_data;
                  iso_hit_position            = iso_position;

                  //vec4 iso_hit_normal_objspc  = compute_iso_normal ( du, dv, dw, ddu, ddv, ddw );
                  iso_hit_normal              = iso_normal;                  
                  ivec2 coords                = ivec2(gl_FragCoord.xy);

                  break;
                }
              }

              last_sample_valid     = true;
              last_sample_data      = sample_data;
              last_sample_position  = sample_position;
              uvw_start             = uvw_next;   // go to next 
            }
  
            // try to intersect included volume
            if ( found_iso_surface ) 
            {
              iso_found = true;
              break;
            } else {
              // if no iso surface found in volume -> set point entry to find next intersection
              t_max     = length(ray_exit.xyz - sample_start.xyz);
              ray_entry = sample_start;
            }
          } // if found an intersection with an obb

          // increase iteration count
          ++inner_ocnode_iterations;
        } // while found obb intersection in ocnode
      } // if ocnode contains potential isovalue
    } // if ocnode contains anything

    // no iso surface was found in this ocnode -> go to next ocnode
    if (iso_found)
    {
      break;
    } else {
      // set ray origin to exit of ocnode 
      ray_entry     = ocnode_exit;
      entry_normal  = exit_normal;
    }
    
    // test if next origin is already out of octree
    vec3 next_ray_entry = ray_entry.xyz + OCTREE_OFFSET_EPSILON * entry_normal.xyz;

    // abort criteria I : iteration if ray exits bounds of octree
    if ( next_ray_entry.x >= octree_bounds_max.x || next_ray_entry.x <= octree_bounds_min.x ||
         next_ray_entry.y >= octree_bounds_max.y || next_ray_entry.y <= octree_bounds_min.y ||
         next_ray_entry.z >= octree_bounds_max.z || next_ray_entry.z <= octree_bounds_min.z ) 
    {
      break;
    }
  

    // abort criteria II : too many iterations
    if ( iters++ > MAX_OCTREE_TRAVERSAL ) 
    {
      break;
    }
  } // while traverse octree

  
  /////////////////////////////////////////////////////////////////////////////
  // shade
  /////////////////////////////////////////////////////////////////////////////
  
  vec4 lightpos           = vec4 ( 0.0, 0.0, 0.0, 1.0); // light from camera
  vec4 pworld             = modelviewmatrix * iso_hit_position;

  iso_hit_normal          = normalmatrix * iso_hit_normal;
  iso_hit_normal          = vec4( faceforward ( normalize(-pworld.xyz), iso_hit_normal.xyz ), 0.0);
 
  if ( iso_found )
  {
    vec3 L        = normalize ( lightpos.xyz - pworld.xyz );
    vec3 N        = normalize ( iso_hit_normal.xyz );
    float diffuse = dot (N , L);
    diffuse       = (diffuse * 0.5f) + 0.5f;

    out_color     = diffuse * vec4((iso_hit_data.xyz - attribute_min.xyz) / (attribute_max.xyz - attribute_min.xyz), 1.0);

    // write result into target texture
    imageStore(colorattachment0, ivec2(gl_FragCoord.xy), out_color );
    float iso_depth = compute_depth ( modelviewmatrix, iso_hit_position, nearplane, farplane );
    imageStore(depthattachment,  ivec2(gl_FragCoord.xy), vec4(iso_depth, iso_depth, iso_depth, 1.0) );

    /////////////////////////////////////////////////////////////////////////////
    // depth correction
    /////////////////////////////////////////////////////////////////////////////
    gl_FragDepth = compute_depth ( pworld, nearplane, farplane );

    // wait until fragments are written
    memoryBarrier();
  }

  

  // do not write into framebuffer
  discard;
}
