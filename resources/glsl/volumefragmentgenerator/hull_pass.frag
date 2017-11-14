/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_multipass/hull_pass.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable
#extension GL_EXT_bindable_uniform : enable
#extension GL_NV_shader_buffer_load : enable

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4                                     modelviewprojectionmatrix;
uniform mat4                                     modelviewmatrix;
uniform mat4                                     modelviewmatrixinverse;
uniform mat4                                     normalmatrix;
                                                 
uniform int                                      width;
uniform int                                      height;
uniform int                                      pagesize;
uniform int                                      pagesize_per_fragment;
uniform float                                    nearplane;
uniform float                                    farplane;
                                                 
uniform ivec2                                    tile_size;

uniform samplerBuffer                            databuffer;

layout (size1x32) coherent uniform uimage2D      index_image;
layout (size1x32) coherent uniform uimage2D      semaphore_image;
layout (size1x32) coherent uniform uimage2D      fragmentcount;
layout (size1x32) coherent uniform uimageBuffer  globalindices;
layout (size1x32) coherent uniform uimageBuffer  master_counter;
layout (size4x32) coherent uniform imageBuffer   fraglist;
layout (size4x32) coherent uniform uimageBuffer  indexlist;

layout(binding = 3, offset = 0) uniform atomic_uint counter;

/********************************************************************************
* input
********************************************************************************/
in vec4        fragposition;
in vec3        parameter;
flat in uvec4  surface_info;
flat in uvec4  volume_info; 

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 color;


/********************************************************************************
* functions
********************************************************************************/
#include "resources/glsl/base/compute_depth.frag"
#include "resources/glsl/base/conversion.glsl"


///////////////////////////////////////////////////////////////////////////////
bool semaphore_acquire ( in ivec2 coords )
{
	return imageAtomicExchange ( semaphore_image, coords, 1U) == 0U;
}

///////////////////////////////////////////////////////////////////////////////
void semaphore_release ( in ivec2 coords )
{
	imageAtomicExchange ( semaphore_image, coords, 0U);
}




/********************************************************************************
* first pass generates depth intervals based on convex hulls of volumes
********************************************************************************/
void main(void)
{
  /*********************************************************************
  * compute fragment data
  *********************************************************************/
  // compute depth as unsigned int or float
  float fdepth = compute_depth ( modelviewmatrix, fragposition, nearplane, farplane ); 

  ivec2 coords     = ivec2(gl_FragCoord.xy);
  bool  loop_abort = false;
  uint chunksize   = pagesize * tile_size.x * tile_size.y;          

  // increase fragment count for fragment
  //uint fragnumber = imageAtomicAdd ( fragmentcount, coords, 1U );

#if 0
  while ( !loop_abort )
  {
    if ( semaphore_acquire ( coords ) )
    {
    	loop_abort = true;

      // get index to fragment list
      uint current_headpointer = imageLoad(index_image, coords).x;
      uint next_headpointer    = current_headpointer + 1U;

      // check if there is still memory in current page
      bool allocate_new_page = (current_headpointer == 0) ||            // no current head pointer
                               (mod(next_headpointer, pagesize) == 0);  // position after head pointer out of page

      // allocate new page in indexbuffer

      // use global counter -> freezes
#if 0
      if ( allocate_new_page ) 
      {
        bool master_accessed = false;
        //while ( !master_accessed ) 
        {
          // try to get access
          //if ( imageAtomicExchange ( master_counter, 1, 1U) == 0U )
          {
            // fragment is processed -> exit next time
            master_accessed = true;
            ivec2 tile_id        = ivec2( mod(coords, tile_size));

            // get next global page and increase next available page by pagesize
            uint next_page_index = imageAtomicAdd ( master_counter, 0, pagesize );
            //uint next_page_index = imageLoad ( master_counter, 0 ).x;
            //imageStore ( master_counter, 0, uvec4(next_page_index + pagesize) );
            
            // return free page and next free page to 
            next_headpointer     = imageAtomicExchange ( globalindices, tile_id.y * tile_size.x + tile_id.x, next_page_index );
            //next_headpointer = imageLoad ( globalindices, tile_id.y * tile_size.x + tile_id.x ).x;
            //imageStore ( globalindices, tile_id.y * tile_size.x + tile_id.x, uvec4(next_page_index) );
            //memoryBarrier();

            //imageAtomicExchange ( master_counter, 1, 0U);
          }
        }
      }
#else
      if ( allocate_new_page ) 
      {
        ivec2 tile_id       = ivec2( mod(coords, tile_size));
        next_headpointer    = imageAtomicAdd ( globalindices, tile_id.y * tile_size.x + tile_id.x, chunksize );
        //next_headpointer    = imageLoad ( globalindices, tile_id.y * tile_size.x + tile_id.x).x;
        //imageStore ( globalindices, tile_id.y * tile_size.x + tile_id.x, uvec4(next_headpointer+chunksize) );
      }
#endif
      // increase write position for current pixel
      imageStore ( index_image, coords, uvec4(next_headpointer) );

      // save index data current fragment
      imageStore ( indexlist, int(next_headpointer), uvec4(current_headpointer, packHalf2x16(parameter.xy), surface_info.x, floatBitsToUint(fdepth) ) );

      // might be necessary to wait until fragment is written
			//memoryBarrier();
			semaphore_release ( coords );
    }
  }
#else

  uint current_headpointer = imageLoad(index_image, coords).x;

  if (current_headpointer == 0) {
    current_headpointer = atomicCounterIncrement(counter);
  }

  uint next_headpointer    = atomicCounterIncrement(counter);

  imageStore ( index_image, coords, uvec4(next_headpointer) );

  imageStore ( indexlist, int(next_headpointer), uvec4(current_headpointer, packHalf2x16(parameter.xy), surface_info.x, floatBitsToUint(fdepth) ) );
#endif


  //memoryBarrier();
  discard;
}

