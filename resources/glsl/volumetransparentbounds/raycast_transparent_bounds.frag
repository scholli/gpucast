/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_multipass/raycast_pass.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable
#extension GL_ARB_conservative_depth : enable

/********************************************************************************
* constants
********************************************************************************/
#define MAX_FLOAT 1E+37

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

layout (size1x32) uniform uimage2D                fragmentcount;
layout (size4x32) uniform imageBuffer             fraglist;
layout (size4x32) uniform uimageBuffer            indexlist;

// data buffer configuration
uniform int                                       pagesize;
uniform int                                       volume_info_offset;
uniform ivec2                                     tile_size;

// parameter related to attached data
uniform vec4                                      global_attribute_min;
uniform vec4                                      global_attribute_max;
uniform vec4                                      threshold;

// render configuration
uniform bool                                      show_isosides;
uniform float                                     newton_epsilon;
uniform int                                       newton_max_iterations;

uniform float                                     surface_transparency;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragment_texcoord;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0)  out vec4    out_color;
layout (depth_less)    out float   gl_FragDepth;

/********************************************************************************
* functions
********************************************************************************/
#include "resources/glsl/base/compute_depth.frag"
#include "resources/glsl/base/faceforward.frag"
#include "resources/glsl/base/phong.frag"
#include "resources/glsl/base/parameter_on_boundary.frag"
#include "resources/glsl/base/conversion.glsl"

#include "resources/glsl/math/raygeneration.glsl.frag"
#include "resources/glsl/math/adjoint.glsl.frag"
#include "resources/glsl/math/euclidian_space.glsl.frag"
#include "resources/glsl/math/horner_surface.glsl.frag"
#include "resources/glsl/math/horner_surface_derivatives.glsl.frag"
#include "resources/glsl/math/horner_volume.glsl.frag"
#include "resources/glsl/math/newton_surface.glsl.frag"
#include "resources/glsl/isosurface/clip_ray_at_nearplane.frag"
#include "resources/glsl/isosurface/target_function.frag"




/********************************************************************************
* raycast pass traverses intervals per fragment and intersects surfaces
********************************************************************************/
void main(void)
{
  /********************************************************************************
  * init output color
  ********************************************************************************/ 
  out_color         = vec4(0.0);
  float out_opacity = 1.0;

  /********************************************************************************
  * retrieve fragment list -> discard empty lists
  ********************************************************************************/ 
  ivec2 coords    = ivec2(gl_FragCoord.xy);

  uint nfragments = imageLoad(fragmentcount, coords).x;
  if ( nfragments == 0 ) 
  {
    discard;
  }

  int start_index = 0;
  
  ivec2 resolution      = ivec2(width, height);
  ivec2 tile_resolution = ivec2(resolution / tile_size) + ivec2(clamp(mod(resolution, tile_size), ivec2(0), ivec2(1)));

  ivec2 tile_id         = ivec2( mod(coords, tile_size));
  ivec2 tile_coords     = coords / tile_size;

  int chunksize         = tile_size.x * tile_size.y * pagesize;
    
  start_index           = tile_coords.y * tile_resolution.x * chunksize + 
                          tile_coords.x * chunksize + 
                          tile_id.y * tile_size.x * pagesize + 
                          tile_id.x * pagesize;

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
    * early ray termination
    ********************************************************************************/ 
    if ( out_opacity < 0.01 )
    {
      break;
    }

    /********************************************************************************
    * retrieve information about fragment, volume and surface
    ********************************************************************************/ 
    uvec4 fragindexinfo       = imageLoad(indexlist, fragindex);
    vec4 fragmentdata0        = imageLoad(fraglist, int(fragindexinfo.z));
    vec4 fragmentdata1        = imageLoad(fraglist, int(fragindexinfo.z + 1));
    vec4 fragposition         = vec4(fragmentdata1.xyz, 1.0);

    vec3 start_uvw            = fragmentdata0.xyz;

    int volume_id             = int(fragindexinfo.y);
    int surface_id            = floatBitsToInt(fragmentdata1.w);

    // unpack float to 4 normalized float and rescale id's to domain [0, 2]
    uvec4 ids                 = intToUInt4(uint(floatBitsToInt(fragmentdata0.w)));

    uint sid                  = uint(ids.x);
    uint tid                  = uint(ids.y);
    uint uid                  = uint(ids.z);

    // flag if surface is in or outside of nurbs volume
    bool is_boundary_surface  = uintToBvec4(ids.w).x;

    // retrieve index of next fragment
    fragindex                 = int(fragindexinfo.x);

    /********************************************************************************
    * ray setup in object coordinates
    ********************************************************************************/ 
    vec4 ray_entry     = fragposition;
    vec4 ray_origin    = modelviewmatrixinverse * vec4(0.0, 0.0, 0.0, 1.0);    
    vec4 ray_direction = vec4(normalize(ray_entry.xyz - ray_origin.xyz), 0.0);

    float fragment_depth = compute_depth ( modelviewmatrix, fragposition, nearplane, farplane );

    /********************************************************************************
    * start ray casting
    ********************************************************************************/ 
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
    bool surface_intersection = newton(uv, newton_epsilon, newton_max_iterations, databuffer, surface_id, order[sid], order[tid], n1, n2, d1, d2, p, du, dv);

    // compute volumetric uvw parameter
    vec3 uvw = vec3(0.0);
    uvw[sid] = uv.x;
    uvw[tid] = uv.y;
    uvw[uid] = clamp(start_uvw[uid], 0.0, 1.0);

#if 1
    if ( is_boundary_surface && surface_intersection )
    {
      found_surface_intersection = true;

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
        rel_attrib = vec4((attrib.xyz - global_attribute_min.xyz) / (global_attribute_max.xyz - global_attribute_min.xyz), 1.0);
      } else {
        if ( target_function(attrib) > target_function(threshold) )
        {
          rel_attrib = vec4(0.1, 0.7, 0.1, 1.0);
        } else {
          rel_attrib = vec4(0.7, 0.1, 0.1, 1.0);
        }
      }

      out_color   += out_opacity * surface_transparency * color * rel_attrib;
      out_opacity *= (1.0 - surface_transparency);
      //return;
    }
#endif

  }

  gl_FragDepth = current_depth;

  //out_color = vec4(float(nfragments)/16.0, 0.0, 0.0, 1.0);
  //out_color  = clamp ( out_color, vec4(0.0), vec4(1.0));

  if (!found_surface_intersection)
  {
    discard;
  }
}

