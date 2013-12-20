/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volumecomposer/raycast_bounds_and_compose_with_isosurface.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable

/********************************************************************************
* constants
********************************************************************************/
#define MAX_UINT          4294967295
#define MAX_INT           2147483647
#define MAX_FLOAT         1E+37
#define EPSILON_FLOAT     0.00001

#define NEWTON_EPSILON    0.001
#define NEWTON_ITERATIONS 6

//#pragma optionNV(fastmath on)
//#pragma optionNV(fastprecision on)
//#pragma optionNV(ifcvt none)
//#pragma optionNV(inline all)
//#pragma optionNV(strict on)
//#pragma optionNV(unroll none)

/********************************************************************************
* uniforms
********************************************************************************/
// screen and camera setup 
uniform int                                       width;
uniform int                                       height;

uniform float                                     nearplane;
uniform float                                     farplane;

uniform mat4                                      modelviewmatrix;
uniform mat4                                      modelviewprojectionmatrix;
uniform mat4                                      modelviewmatrixinverse;
uniform mat4                                      normalmatrix;

// data buffer
uniform samplerBuffer                             databuffer;
uniform samplerBuffer                             attributebuffer;
uniform sampler2D                                 transfertexture;

layout (size1x32) uniform uimageBuffer            globalindices;
layout (size1x32) uniform uimage2D                index_image;
layout (size1x32) uniform uimage2D                fragmentcount;
layout (size4x32) uniform imageBuffer             fraglist;
layout (size4x32) uniform uimageBuffer            indexlist;

uniform sampler2D                                 isosurface_color;
uniform sampler2D                                 isosurface_depth;

// data buffer configuration
uniform int                                       pagesize;
uniform int                                       volume_info_offset;

// parameter related to attached data
uniform vec4                                      global_attribute_min;
uniform vec4                                      global_attribute_max;
uniform vec4                                      threshold;

// render configuration
uniform bool                                      adaptive_sampling;
uniform bool                                      show_isosides;
uniform float                                     min_sample_distance;
uniform float                                     max_sample_distance;
uniform float                                     adaptive_sample_scale;
uniform float                                     newton_epsilon;
uniform int                                       newton_max_iterations;
uniform int                                       max_steps_binary_search;

uniform float                                     surface_transparency;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragment_texcoord;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 out_color;

/********************************************************************************
* functions
********************************************************************************/
#include "./libgpucast/glsl/base/compute_depth.frag"
#include "./libgpucast/glsl/base/faceforward.frag"
#include "./libgpucast/glsl/base/phong.frag"
#include "./libgpucast/glsl/base/conversion.frag"

#include "./libgpucast/glsl/math/raygeneration.glsl.frag"
#include "./libgpucast/glsl/math/adjoint.glsl.frag"
#include "./libgpucast/glsl/math/euclidian_space.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./libgpucast/glsl/math/horner_volume.glsl.frag"
#include "./libgpucast/glsl/math/newton_surface.glsl.frag"

#include "./libgpucast/glsl/isosurface/clip_ray_at_nearplane.frag"
#include "./libgpucast/glsl/isosurface/target_function.frag"


///////////////////////////////////////////////////////////////////////////////
bool parameter_on_domain_boundary ( in vec3 uvw, 
                                    in vec3 uvwmin_local,
                                    in vec3 uvwmax_local,
                                    in vec3 uvwmin_global,
                                    in vec3 uvwmax_global,
                                    in float epsilon )
{
  vec3 uvw_global = uvwmin_local + uvw * (uvwmax_local - uvwmin_local);

  return uvw_global.x > uvwmax_global.x - epsilon ||
         uvw_global.y > uvwmax_global.y - epsilon ||
         uvw_global.z > uvwmax_global.z - epsilon ||
         uvw_global.x < uvwmin_global.x + epsilon ||
         uvw_global.y < uvwmin_global.y + epsilon ||
         uvw_global.z < uvwmin_global.z + epsilon;
}


vec4 attribute_shading ( in sampler2D   transfertexture,
                         in vec4        attribute_value,
                         in vec4        attribute_min,
                         in vec4        attribute_max )
{
  vec4 normalized = vec4((attribute_value.xyz - attribute_min.xyz) / (attribute_max.xyz - attribute_min.xyz), 1.0);
  //return texture(transfertexture, 0.4);
  return normalized;
}



/********************************************************************************
* raycast pass traverses intervals per fragment and intersects surfaces
********************************************************************************/
void main(void)
{
  /********************************************************************************
  * init output color
  ********************************************************************************/ 
  out_color = vec4(0.0);

  vec4  isocolor = texture(isosurface_color, gl_FragCoord.xy / vec2(width,height) );
  float isodepth = texture(isosurface_depth, gl_FragCoord.xy / vec2(width,height) ).x;

  float opacity  = 1.0;

  /********************************************************************************
  * retrieve fragment list -> discard empty lists
  ********************************************************************************/ 
  ivec2 coords    = ivec2(gl_FragCoord.xy);

  uint nfragments = imageLoad(fragmentcount, coords).x;
  if ( nfragments == 0 ) 
  {
    discard;
  }

  int start_index = pagesize * coords.x + pagesize * coords.y * width;
  int fragindex   = start_index;

  /********************************************************************************
  * allocate variables for ray casting result
  ********************************************************************************/ 
  bool  found_surface_intersection  = false;
  bool  found_iso_surface           = false;

  vec3  iso_surface_uvw             = vec3(0.0);
  vec4  iso_surface_position        = vec4(0.0);
  vec4  iso_surface_normal          = vec4(0.0);
  vec4  iso_surface_attribute       = vec4(0.0);

  float current_depth               = MAX_FLOAT;
  vec4  last_sample_position        = vec4(0.0);
  vec4  attrib, adu, adv, adw;
  vec4  p, du, dv, dw;

  /********************************************************************************
  * traverse depth-sorted fragments
  ********************************************************************************/ 
  for ( int i = 0; i != nfragments; ++i )
  {
    /********************************************************************************
    * retrieve information about fragment, volume and surface
    ********************************************************************************/ 
    uvec4 fragindexinfo = imageLoad(indexlist, fragindex);
    vec4 fragmentdata0  = imageLoad(fraglist, int(fragindexinfo.z));
    vec4 fragmentdata1  = imageLoad(fraglist, int(fragindexinfo.z + 1));
    vec4 fragposition   = vec4(fragmentdata1.xyz, 1.0);

    vec3 start_uvw      = fragmentdata0.xyz;

    int volume_id       = int(fragindexinfo.y);
    int surface_id      = floatBitsToInt(fragmentdata1.w);

    // unpack float to 4 normalized float and rescale id's to domain [0, 2]
    uvec4 ids           = intToUInt4(uint(floatBitsToInt(fragmentdata0.w)));
    uint sid            = ids.x;
    uint tid            = ids.y;
    uint uid            = ids.z;

    // retrieve index of next fragment
    fragindex           = int(fragindexinfo.x);

    /********************************************************************************
    * ray setup in object coordinates
    ********************************************************************************/ 
    vec4 ray_entry     = fragposition;
    vec4 ray_origin    = modelviewmatrixinverse * vec4(0.0, 0.0, 0.0, 1.0);    
    vec4 ray_direction = vec4(normalize(ray_entry.xyz - ray_origin.xyz), 0.0);

    /********************************************************************************
    * early ray termination
    ********************************************************************************/ 
    float fragment_depth = compute_depth ( modelviewmatrix, fragposition, nearplane, farplane );

    // terminate raycasting if fragment's depth is greater than found intersection
    if ( opacity < 0.01 )
    {
      break;
    }

    if ( isocolor.w > 0.0 && out_color.w > 0.0 ) // nasty hack -> temporary
    {
      out_color        += opacity * isocolor;
      opacity           = 0.0;
      gl_FragDepth      = isodepth;
    }

    /********************************************************************************
    * start analyzing 
    ********************************************************************************/ 
    // debug: pick a certain fragment
    //if ( i == nfragments-1 )
    //if ( i == 0 )
    //if ( volume_id == 7 * 87 )
    //{
      vec4 volume_info    = texelFetchBuffer( databuffer, volume_id     );
      vec4 volume_info2   = texelFetchBuffer( databuffer, volume_id + 1 );

      int attribute_id    = int(volume_info.z);
      float bbox_size     = volume_info.w;

      /********************************************************************************
      * ray setup 
      ********************************************************************************/ 
      vec3  n1    = vec3(0.0);
      vec3  n2    = vec3(0.0);
      float d1    = 0.0;
      float d2    = 0.0;

      ivec3 order = ivec3(int(volume_info2.x), int(volume_info2.y), int(volume_info2.z));
      vec2  uv    = vec2(start_uvw[sid], start_uvw[tid]);

      raygen ( fragposition, modelviewmatrixinverse, n1, n2, d1, d2 );

      /********************************************************************************
      * intersect surface
      ********************************************************************************/ 
      bool surface_intersection = newton(uv, NEWTON_EPSILON, NEWTON_ITERATIONS, databuffer, surface_id, order[sid], order[tid], n1, n2, d1, d2, p, du, dv);

      // retrieve NURBS domain to determine if surface is a  boundarysurface
      vec3 uvwmin_local   = texelFetchBuffer( databuffer, volume_id + 2 ).xyz;
      vec3 uvwmax_local   = texelFetchBuffer( databuffer, volume_id + 3 ).xyz;
      vec3 uvwmin_global  = texelFetchBuffer( databuffer, volume_id + 4 ).xyz;
      vec3 uvwmax_global  = texelFetchBuffer( databuffer, volume_id + 5 ).xyz;

      // compute volumetric uvw parameter
      vec3 uvw = vec3(0.0);
      uvw[sid] = uv.x;
      uvw[tid] = uv.y;
      uvw[uid] = clamp(start_uvw[uid], 0.0, 1.0);
      
      bool is_boundary_surface  = parameter_on_domain_boundary ( uvw, uvwmin_local, uvwmax_local, uvwmin_global, uvwmax_global, EPSILON_FLOAT );
#if 1
      if ( is_boundary_surface && surface_intersection )
      {
        found_surface_intersection    = true;

        /********************************************************************************
        * depth correction
        ********************************************************************************/ 
        vec4  pview               = modelviewmatrix * vec4(p.xyz, 1.0);  
        float intersection_depth  = compute_depth ( pview, nearplane, farplane ); 
        vec4  normal              = normalmatrix * vec4( cross( normalize(du.xyz), normalize(dv.xyz)), 0.0);

        if ( intersection_depth < current_depth )
        {
          current_depth     = intersection_depth;
          gl_FragDepth      = intersection_depth;
        }

        /********************************************************************************
        * shade surface hit
        ********************************************************************************/ 
        vec4 lightpos = vec4 ( 0.0, 0.0, 0.0, 1.0); // light from camera
        vec4 color    = phong_shading ( pview, normal, lightpos );

        evaluateVolume(attributebuffer, attribute_id + volume_info_offset, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, attrib, adu, adv, adw);  

        vec4 rel_attrib = vec4(0.0);
        if ( !show_isosides )
        {
          rel_attrib = attribute_shading ( transfertexture, attrib, global_attribute_min, global_attribute_max );
        } else {
          if ( target_function(attrib) > target_function(threshold) )
          {
            rel_attrib = vec4(0.1, 0.7, 0.1, surface_transparency);
          } else {
            rel_attrib = vec4(0.7, 0.1, 0.1, surface_transparency);
          }
        }

        color   *= rel_attrib;
        color.w  = 1.0;

        opacity          *= (1.0 - surface_transparency);
        out_color        += opacity * color;
      }
#endif
      /********************************************************************************
      * raycast for isosurface
      ********************************************************************************/ 
#if 0
      if ( surface_intersection ) 
      {
        // store positive result of surface intersection
        found_surface_intersection = true;

        vec4 local_attribute_min = texelFetchBuffer(attributebuffer, attribute_id    );
        vec4 local_attribute_max = texelFetchBuffer(attributebuffer, attribute_id + 1);

        //bool volume_includes_potential_iso_value = false;
        bool volume_includes_potential_iso_value = is_in_range ( threshold, local_attribute_min, local_attribute_max );

        // if volume includes a potential iso surface and no iso surface was found yet
        if ( volume_includes_potential_iso_value ) 
        {
          /********************************************************************************
          * search volume for iso_surface
          ********************************************************************************/ 

          found_iso_surface = search_range_for_iso_surface ( databuffer,
                                                             attributebuffer,
                                                             volume_id,
                                                             attribute_id,
                                                             pagesize_volumeinfo,
                                                             order,
                                                             uvw,
                                                             n1, n2, d1, d2,
                                                             ray_origin,
                                                             ray_direction,
                                                             bbox_size,
                                                             min_sample_distance,
                                                             max_sample_distance,
                                                             adaptive_sample_scale,
                                                             threshold,
                                                             iso_surface_uvw,
                                                             iso_surface_position,
                                                             iso_surface_attribute,
                                                             iso_surface_normal );

          /*evaluateVolume ( databuffer, volume_id + pagesize_volumeinfo, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, point, pdu, pdv, pdw);  
          evaluateVolume ( attributebuffer, attribute_id + pagesize_volumeinfo, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, attrib, adu, adv, adw);  
          iso_surface_normal = compute_iso_normal ( pdu, pdv, pdw, adu, adv, adw );*/

          //found_iso_surface = true;
          if ( found_iso_surface )
          {
            /*vec4 lightpos = vec4 ( 0.0, 0.0, 0.0, 1.0); // light from camera
            vec4 pview    = modelviewmatrix * vec4(iso_surface_position.xyz, 1.0);  
            vec3 N        = (normalmatrix * iso_surface_normal).xyz;
            N             = myfaceforward ( -pview.xyz, N );
            vec4 color    = phong_shading ( pview, vec4(N, 0.0), lightpos );
            vec4 rel_attribute = (iso_surface_attribute - global_attribute_min ) / (global_attribute_max - global_attribute_min);*/
            break;
            
            //out_color = vec4(1.0, 0.0, 0.0, 1.0);
          }
        }

        ////////////////////////////////////
        // start iso search within interval
        //////////////////////////////////// 
      } else {
        // do nothing -> volume wasn't intersected
      }
#endif
  }

  //out_color /= out_color.w;
  out_color  = clamp ( out_color, vec4(0.0), vec4(1.0));

  if (!found_surface_intersection)
  {
    discard;
  }
}

