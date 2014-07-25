#ifndef LIBGPUCAST_VERIFY_DEPTH_ORDER_H
#define LIBGPUCAST_VERIFY_DEPTH_ORDER_H

///////////////////////////////////////////////////////////////////////////////
__device__ inline bool
verify_depth_order ( uint4*  indexlist,
                     int     current_fragment_index,
                     float&  nfragments )
{
  float depth          = -1.0f;

  while ( current_fragment_index != 0 )
  {
    // get fragment information
    uint4 current_fragment = indexlist[current_fragment_index]; 
    float current_depth    = intBitsToFloat(current_fragment.w);
    if ( depth > current_depth ) {
      return false;
    } else {
      depth = current_depth;
    }
    current_fragment_index = int(current_fragment.x); 
    nfragments += 0.05;
  }
  return true;
}

#endif

