/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_multipass/sort_pass.frag
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

/********************************************************************************
* uniforms
********************************************************************************/
uniform int                                       width;
uniform int                                       height;
uniform int                                       pagesize;
uniform ivec2                                     tile_size;

layout (size1x32) uniform uimage2D                fragmentcount;
layout (size1x32) uniform uimage2D                indeximage;
layout (size4x32) uniform uimageBuffer            indexlist;

/********************************************************************************
* input
********************************************************************************/

/********************************************************************************
* output
********************************************************************************/

/********************************************************************************
* functions
********************************************************************************/

uint count_fragments ( in uint start_index )
{
  uint current_index = start_index;
  uint nfragments    = 0;

  uvec4 entry        = imageLoad ( indexlist, int(current_index) );
  while ( entry.x != 0 )
  {
    ++nfragments;
    current_index = entry.x;
    entry         = imageLoad ( indexlist, int(current_index) );
  }

  return nfragments;
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

      if ( intBitsToFloat(int(entry0.w)) > intBitsToFloat(int(entry1.w)) ) 
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
* sort pass for fragment sorting
********************************************************************************/
void main(void)
{
  /********************************************************************************
  * retrieve headpointer -> discard empty lists
  ********************************************************************************/ 
  ivec2 coords    = ivec2(gl_FragCoord.xy);

  uint headpointer = imageLoad(indeximage, coords).x;

  if ( headpointer == 0 ) 
  {
    discard;
  } else {

    /********************************************************************************
    * sort fragment list 
    ********************************************************************************/ 
#if 0 
    uint nfragments = count_fragments ( headpointer );
#else
    uint nfragments = imageLoad ( fragmentcount, coords).x;
#endif

    bubble_sort ( int(headpointer), nfragments );

    discard;
  }
}

