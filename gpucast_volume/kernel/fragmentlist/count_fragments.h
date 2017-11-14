#ifndef LIBGPUCAST_COUNT_FRAGMENTS_H
#define LIBGPUCAST_COUNT_FRAGMENTS_H

#include "math/conversion.h"

///////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned 
count_fragments ( uint4*   indexlist,
                  unsigned start_index )
{
  if ( start_index == 0 ) {
    return 0;
  }

  unsigned next_entry = start_index;
  unsigned fragments  = 1;

  uint4 entry = indexlist[next_entry]; 
  while ( entry.x != 0 )
  {
    ++fragments;
    next_entry  = entry.x;
    entry       = indexlist[next_entry]; 

    // critical abort
    if (fragments > 100) {
      break;
    }
  }

  return fragments;
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned 
count_intersections ( uint4*   indexlist,
                      float4*  fraglist,
                      unsigned start_index )
{
  unsigned next_entry = start_index;
  unsigned fragments  = 0;

  uint4 entry = indexlist[next_entry]; 
  while ( entry.x != 0 )
  {
    unsigned volume_info_index = entry.z;
    if ( floatBitsToInt(fraglist[volume_info_index + 1].w) > 0 ) 
    //if ( volume_info_index > 0 )
    {
      ++fragments;
    }

    next_entry  = entry.x;
    entry       = indexlist[next_entry]; 
  }

  return fragments;
}

#endif

