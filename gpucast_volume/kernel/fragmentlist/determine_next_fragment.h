#ifndef LIBGPUCAST_DETERMINE_NEXT_FRAGMENT_H
#define LIBGPUCAST_DETERMINE_NEXT_FRAGMENT_H

///////////////////////////////////////////////////////////////////////////////
__device__ inline bool
determine_next_fragment ( uint4*  indexlist,
                          int&    current_fragment_index, 
                          float   current_fragment_depth,
                          uint4&  next_fragment_data )
{
  // while there is a valid next fragment 
  while ( current_fragment_index != 0 )
  {
    // get fragment information
    next_fragment_data = indexlist[current_fragment_index]; 

    // if fragment's depth is behind current depth -> stop traversal
    if ( current_fragment_depth < intBitsToFloat(next_fragment_data.w) )
    { 
      current_fragment_index = int(next_fragment_data.x);
      return true;
    } else { // go to next fragment
      current_fragment_index  = int(next_fragment_data.x);
    }
  }
  return false;
}

#endif

