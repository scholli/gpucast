/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volumefraglistraycasting/fraglist_raycasting.frag
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

// volume data
uniform samplerBuffer                             pointbuffer;
uniform samplerBuffer                             attributebuffer;

// data buffer configuration
uniform int                                       volume_info_offset;

// fragment data
layout (size4x32) uniform imageBuffer             fraglist;
layout (size4x32) uniform uimageBuffer            indexlist;

// fragment data configuration
uniform int                                       pagesize;
uniform int                                       pagesize_volumeinfo;
uniform ivec2                                     tile_size;

// parameter related to attached data
uniform vec4                                      threshold;
uniform float                                     surface_transparency;

// parametrization
uniform bool                                      adaptive_newton_epsilon;
uniform float                                     fixed_newton_epsilon;
uniform int                                       newton_max_iterations;
uniform bool                                      backface_culling;


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
#include "./libgpucast/glsl/base/parameter_on_boundary.frag"
#include "./libgpucast/glsl/base/conversion.frag"

#include "./libgpucast/glsl/math/raygeneration.glsl.frag"
#include "./libgpucast/glsl/math/adjoint.glsl.frag"
#include "./libgpucast/glsl/math/euclidian_space.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./libgpucast/glsl/math/newton_surface.glsl.frag"

#include "./libgpucast/glsl/isosurface/clip_ray_at_nearplane.frag"
#include "./libgpucast/glsl/isosurface/target_function.frag"
#include "./libgpucast/glsl/isosurface/in_bezier_domain.frag"
#include "./libgpucast/glsl/isosurface/point_ray_distance.frag"

///////////////////////////////////////////////////////////////////////////////
int get_fragment_count ( in int start_index )
{
  int next_entry = start_index;
  int fragments  = 0;

  uvec4 entry = imageLoad ( indexlist, next_entry ); 
  while ( entry.x != 0 )
  {
    ++fragments;
    next_entry = int(entry.x);
    entry = imageLoad ( indexlist, next_entry ); 
  }

  return fragments;
}

///////////////////////////////////////////////////////////////////////////////
void bubble_sort ( in int start_index,
                   in uint nfragments )
{
  // sort list of fragments
  for ( int i = 0; i != nfragments; ++i )
  { 
    int index   = start_index;

    for ( int j = 0; j != int(nfragments - 1); ++j )
    { 
      // get entry for this fragment
      uvec4 entry0 = imageLoad ( indexlist, index );
      uvec4 entry1 = imageLoad ( indexlist, int(entry0.x) );
#if 0
      if ( entry0.w > entry1.w //|| 
         // (fabs(entry0.w - entry1.w) < 0.0001f && entry0.y < entry1.y) 
         ) 
#else
      if ( intBitsToFloat(int(entry0.w)) > intBitsToFloat(int(entry1.w)) 
         ) 
#endif
      {
        // swizzle depth and related data
        imageStore(indexlist, index,         uvec4(entry0.x, entry1.yzw)); 
        imageStore(indexlist, int(entry0.x), uvec4(entry1.x, entry0.yzw));
      }

      index = int(entry0.x);
    }
  }
}


/********************************************************************************
* raycast pass traverses intervals per fragment and intersects surfaces
********************************************************************************/
void main(void)
{
  /********************************************************************************
  * retrieve fragment list -> discard empty lists
  ********************************************************************************/ 
  int start_index = 0;
  ivec2 coords            = ivec2(gl_FragCoord.xy);

  ivec2 resolution      = ivec2(width, height);
  ivec2 tile_resolution = ivec2(resolution / tile_size) + ivec2(clamp(mod(resolution, tile_size), ivec2(0), ivec2(1)));

  ivec2 tile_id         = ivec2( mod(coords, tile_size));
  ivec2 tile_coords     = coords / tile_size;

  int chunksize         = tile_size.x * tile_size.y * pagesize;

  start_index           = tile_coords.y * tile_resolution.x * chunksize + 
                          tile_coords.x * chunksize + 
                          tile_id.y * tile_size.x * pagesize + 
                          tile_id.x * pagesize;

  int nfragments          = get_fragment_count ( start_index );

  if ( nfragments == 0 ) 
  {
    discard;
  }

  /********************************************************************************
  * allocate variables for ray casting result
  ********************************************************************************/ 
  int   fragindex                   = start_index;
  bool  found_surface_intersection  = false;
  ivec4 found_surface_intersections = ivec4(-1);
  bool  found_iso_surface           = false;

  vec3  iso_surface_uvw             = vec3(0.0);
  vec4  iso_surface_position        = vec4(0.0);
  vec4  iso_surface_normal          = vec4(0.0);
  vec4  iso_surface_attribute       = vec4(0.0);

  float current_depth               = MAX_FLOAT;
  vec4  last_sample_position        = vec4(0.0);

  /********************************************************************************
  * init out color and transparency
  ********************************************************************************/ 
  out_color                          = vec4(0.0);
  float out_opacity                  = 1.0;
  float isosurface_opacity           = surface_transparency;
  
  /********************************************************************************
  * traverse depth-sorted fragments
  ********************************************************************************/ 
  for ( int i = 0; i != nfragments; ++i )
  {
    /********************************************************************************
    * quit shading when alpha=1.0 is reached
    ********************************************************************************/ 
    if ( out_opacity < 0.01 )
    {
      break;
    }

    /********************************************************************************
    * retrieve information about fragment, volume and surface
    ********************************************************************************/ 
    uvec4 fragindexinfo        = imageLoad(indexlist, fragindex);
    vec4  fragmentdata0        = imageLoad(fraglist, int(fragindexinfo.z));
    vec4  fragmentdata1        = imageLoad(fraglist, int(fragindexinfo.z + 1));
    vec4  fragmentdata3        = imageLoad(fraglist, int(fragindexinfo.z + 3));
    vec4  fragposition         = vec4(fragmentdata1.xyz, 1.0);
                               
    vec3  start_uvw            = fragmentdata0.xyz;
                               
    int   volume_id            = int(fragindexinfo.y);
    int   surface_id           = floatBitsToInt(fragmentdata1.w);

    vec4  volume_info          = texelFetchBuffer( pointbuffer, volume_id + 1 );
    ivec3 order                = ivec3(volume_info.xyz);

    uvec4 ids                  = intToUInt4(uint(floatBitsToInt(fragmentdata0.w)));                           
    uint  sid                  = ids.x;
    uint  tid                  = ids.y;
    uint  uid                  = ids.z;

    bool  is_outer_surface     = uintToBvec4(ids.w).x;
    bool  contains_isosurface  = uintToBvec4(ids.w).y;

    vec2  uv                   = vec2(start_uvw[sid], start_uvw[tid]);

    /********************************************************************************
    * ray setup
    ********************************************************************************/ 
    vec3  n1    = vec3(0.0);
    vec3  n2    = vec3(0.0);
    float d1    = 0.0;
    float d2    = 0.0;

    raygen ( fragposition, modelviewmatrixinverse, n1, n2, d1, d2 );

    /********************************************************************************
    * intersect surface
    ********************************************************************************/ 
    float newton_epsilon = fixed_newton_epsilon;
    vec4 p, du, dv;
    bool surface_intersection = newton(uv, newton_epsilon, newton_max_iterations, pointbuffer, surface_id, order[sid], order[tid], n1, n2, d1, d2, p, du, dv);

    // compute volumetric uvw parameter
    vec3 uvw = vec3(0.0);
    uvw[sid] = uv.x;
    uvw[tid] = uv.y;
    uvw[uid] = clamp(start_uvw[uid], 0.0, 1.0);

    float fdepth = compute_depth ( modelviewmatrix, p, nearplane, farplane ); 
    uint  udepth = uint(fdepth * MAX_UINT);

    if ( surface_intersection )
    {
      // update intersection depth
#if 0
      imageStore(indexlist, int(fragindex), uvec4(fragindexinfo.xyz, udepth) );
#else 
      imageStore(indexlist, int(fragindex), uvec4(fragindexinfo.xyz, floatBitsToInt(fdepth)) );
#endif

      // update uvw parameter of real intersection
      imageStore(fraglist,  int(fragindexinfo.z    ), vec4(uvw, fragmentdata0.w) );      
    }

    // store information about intersection 
    // todo: assuming that this data is a placeholder it might be useful to write it only IF there is an intersection
    vec4 intersection_data0 = vec4(float(surface_intersection), p.xyz);
    vec4 intersection_data1 = vec4(normalize(cross(du.xyz,dv.xyz)), fragmentdata3.w); // intersection normal, unique_surface_identifier 
    imageStore(fraglist, int(fragindexinfo.z + 2), intersection_data0);
    imageStore(fraglist, int(fragindexinfo.z + 3), intersection_data1);
    imageStore(fraglist, int(fragindexinfo.z + 5), vec4(uvw,0.0) );

    // go to next fragment
    fragindex = int(fragindexinfo.x);
  }

  // sort fragments by intersection depth
  bubble_sort(start_index, uint(nfragments));

  discard;
}





