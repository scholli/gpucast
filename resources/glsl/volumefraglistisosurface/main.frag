/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volumefraglistraycasting/main.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
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

// volume data
uniform samplerBuffer                             databuffer;
uniform samplerBuffer                             attributebuffer;

// data buffer configuration
uniform int                                       volume_info_offset;

// fragment data
layout (size1x32) uniform uimageBuffer            globalindices;
layout (size1x32) uniform uimage2D                index_image;
layout (size1x32) uniform uimage2D                fragmentcount;
layout (size4x32) uniform imageBuffer             fraglist;
layout (size4x32) uniform uimageBuffer            indexlist;

// fragment data configuration
uniform int                                       pagesize;
uniform int                                       pagesize_volumeinfo;
uniform ivec2                                     tile_size;

// parameter related to attached data
uniform vec4                                      global_attribute_min;
uniform vec4                                      global_attribute_max;
uniform vec4                                      threshold;
uniform float                                     surface_transparency;
uniform bool                                      show_isosides;

// parametrization
uniform bool                                      adaptive_newton;
uniform float                                     fixed_newton_epsilon;
uniform int                                       newton_max_iterations;
uniform bool                                      adaptive_sampling;
uniform bool                                      adaptive_newton_epsilon;
uniform bool                                      backface_culling;
uniform float                                     min_sample_distance;
uniform float                                     max_sample_distance;
uniform float                                     adaptive_sample_scale;
uniform int                                       max_steps_binary_search;


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
#include "resources/glsl/math/newton_volume.glsl.frag"

#include "resources/glsl/isosurface/clip_ray_at_nearplane.frag"
#include "resources/glsl/isosurface/target_function.frag"
#include "resources/glsl/isosurface/compute_iso_normal.frag"
#include "resources/glsl/isosurface/compute_next_sampling_position.frag"
#include "resources/glsl/isosurface/binary_search_for_isosurface.frag"
#include "resources/glsl/isosurface/in_bezier_domain.frag"
#include "resources/glsl/isosurface/point_ray_distance.frag"
#include "resources/glsl/isosurface/validate_isosurface_intersection.frag"
#include "resources/glsl/isosurface/search_volume_for_isosurface.frag"

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
bool search_volume_exit_fragment ( in uint           volume_id, 
                                   in uint           fragindex, 
                                   in int            remaining_fragments,
                                   out uvec4         exit_fragment,
                                   out uint          nfragments_found)
{
  bool exit_found  = false;
  nfragments_found = 0;

  for (int i = 0; i < remaining_fragments; ++i)
  {
    uvec4 fraginfo = imageLoad(indexlist, int(fragindex));
    fragindex      = fraginfo.x;

    if ( fraginfo.y == volume_id )
    {
      exit_found    = true;
      exit_fragment = fraginfo;
      ++nfragments_found;
    }
  }

  return exit_found;
}

///////////////////////////////////////////////////////////////////////////////
bool in_domain (in vec3 p, in vec3 pmin, in vec3 pmax)
{
  return  p.x >= pmin.x && 
          p.y >= pmin.y && 
          p.z >= pmin.z && 
          p.x <= pmax.x && 
          p.y <= pmax.y && 
          p.z <= pmax.z; 
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
  float  isosurface_opacity          = surface_transparency;
  
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
    vec4 fragmentdata0         = imageLoad(fraglist, int(fragindexinfo.z));
    vec4 fragmentdata1         = imageLoad(fraglist, int(fragindexinfo.z + 1));
    vec4 fragposition          = vec4(fragmentdata1.xyz, 1.0);
                               
    vec3 start_uvw             = fragmentdata0.xyz;
                               
    int volume_id              = int(fragindexinfo.y);
    int surface_id             = floatBitsToInt(fragmentdata1.w);

    vec4 volume_info           = texelFetchBuffer( databuffer, volume_id     );
    vec4 volume_info2          = texelFetchBuffer( databuffer, volume_id + 1 );
                               
    int attribute_id           = int(volume_info.z);
    float volume_bbox_diameter = volume_info.w;
    ivec3 order                = ivec3(int(volume_info2.x), int(volume_info2.y), int(volume_info2.z));
    vec3  uvw                  = start_uvw;

    /********************************************************************************
    * determine if ray enters volume at this fragment and search for exit fragment
    ********************************************************************************/ 
    uvec4 fragment_volume_exit  = uvec4(0);
    uint  volume_fragment_count = 0;
    bool iso_surface_search_necessary = search_volume_exit_fragment ( volume_id, 
                                                                      fragindexinfo.x, 
                                                                      nfragments - i - 1 ,
                                                                      fragment_volume_exit,
                                                                      volume_fragment_count );

    /********************************************************************************
    * ray setup
    ********************************************************************************/ 
    vec4 ray_entry     = fragposition;
    vec4 ray_exit      = imageLoad(fraglist, int(fragment_volume_exit.z + 1));

    /********************************************************************************
    * raycast for iso surface
    ********************************************************************************/ 

    // search volume for iso surface
    if ( iso_surface_search_necessary ) 
    {

      bool continue_isosurface_search                   = true;
      int isosurface_intersections_per_volume           = 0;
      int max_isosurface_intersections_per_volume       = 3;

      while ( continue_isosurface_search && isosurface_intersections_per_volume < max_isosurface_intersections_per_volume )
      {
        ++isosurface_intersections_per_volume;

        continue_isosurface_search = search_volume_for_iso_surface ( databuffer,
                                                                     attributebuffer,
                                                                     volume_id + volume_info_offset,
                                                                     attribute_id + volume_info_offset,
                                                                     uvec3(order),
                                                                     start_uvw,
                                                                     threshold,
                                                                     ray_entry,
                                                                     ray_exit,
                                                                     adaptive_sampling,
                                                                     volume_bbox_diameter,
                                                                     min_sample_distance,
                                                                     max_sample_distance,
                                                                     adaptive_sample_scale,
                                                                     adaptive_newton,
                                                                     fixed_newton_epsilon,
                                                                     newton_max_iterations,
                                                                     max_steps_binary_search,
                                                                     iso_surface_position,
                                                                     iso_surface_attribute,
                                                                     iso_surface_normal,
                                                                     iso_surface_uvw,
                                                                     ray_entry,
                                                                     start_uvw);
        if ( continue_isosurface_search )
        {
          found_iso_surface  = true;

          // shade
          vec4 lightpos      = vec4(0.0f, 0.0f, 0.0f, 1.0f); // light from camera
          vec4 pworld        = modelviewmatrix * iso_surface_position;
          iso_surface_normal = normalmatrix * iso_surface_normal;

          current_depth      = compute_depth ( pworld, nearplane, farplane );
          vec3 L             = normalize ( lightpos.xyz - pworld.xyz );
          vec3 N             = normalize ( iso_surface_normal.xyz );
          N                  = faceforward ( -pworld.xyz, N );
          float diffuse      = dot (N , L);
          diffuse            = ( diffuse * 0.5f ) + 0.5f;

          out_color         += isosurface_opacity * out_opacity * diffuse * vec4((iso_surface_attribute.xyz - global_attribute_min.xyz) / (global_attribute_max.xyz - global_attribute_min.xyz), 1.0);
          out_opacity       *= (1.0f - isosurface_opacity);
        }
      } // while continue search for isosurface intersections (within one volume)
    } // found_exit fragment -> search interval for iso surface

    // go to next fragment
    fragindex           = int(fragindexinfo.x);
  } // for all fraglments

  if (!found_iso_surface)
  {
    discard;
  } 
}





