/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_multipass/clean_pass.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#version 420 core
#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_shader_image_load_store : enable

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4          modelviewprojectionmatrix;
uniform mat4          modelviewmatrix;
uniform mat4          modelviewmatrixinverse;
uniform mat4          normalmatrix;

uniform int           width;
uniform int           height;
uniform int           pagesize;
uniform int           pagesize_per_fragment;

uniform ivec2         tile_size;

layout (size1x32) coherent uniform uimage2D      index_image;
layout (size1x32) coherent uniform uimage2D      semaphore_image;
layout (size1x32) coherent uniform uimage2D      fragmentcount;
layout (size1x32) coherent uniform uimageBuffer  globalindices;
layout (size1x32) coherent uniform uimageBuffer  master_counter;
layout (size4x32) coherent uniform uimageBuffer  indexlist;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragment_texcoord;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 color;

/********************************************************************************
* functions
********************************************************************************/

/********************************************************************************
* clean pass cleans index image texture
********************************************************************************/
void main(void)
{
  ivec2 coords          = ivec2(gl_FragCoord.xy);

  ivec2 resolution      = ivec2(width, height);
  ivec2 tile_resolution = ivec2(resolution / tile_size) + ivec2(clamp(mod(resolution, tile_size), ivec2(0), ivec2(1)));

  ivec2 tile_id         = ivec2( mod(coords, tile_size));
  ivec2 tile_coords     = coords / tile_size;

  int chunksize         = tile_size.x * tile_size.y * pagesize;

  int index             = 0;

  imageStore ( index_image,     coords, uvec4(index) );
  imageStore ( semaphore_image, coords, uvec4(0U) );
  imageStore ( fragmentcount,   coords, uvec4(0U) );
  imageStore ( indexlist,       index,  uvec4(0U, 0U, 0U, 0U) );
    
  if ( coords.x < tile_size.x && coords.y < tile_size.y )
  {
    // headpointer 0 -> nullpointer ==> first heaedpointer = 1
    uint first_page_index = 1;

	  uint next_free_index = first_page_index + 
                           tile_id.y * tile_size.x * pagesize + 
                           tile_id.x * pagesize; 

	  imageStore ( globalindices, tile_id.y * tile_size.x + tile_id.x,      uvec4(next_free_index) );
  }

// global counter
#if 0
  if ( coords.x == 0 && coords.y == 0 )
  {
    imageStore ( master_counter, 0, uvec4(0U) );
    imageStore ( master_counter, 1, uvec4(0U) );
  }
#endif

  discard; // do not write into framebuffer
}

