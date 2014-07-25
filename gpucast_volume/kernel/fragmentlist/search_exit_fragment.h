#ifndef LIBGPUCAST_SEARCH_EXIT_FRAGMENT_H
#define LIBGPUCAST_SEARCH_EXIT_FRAGMENT_H

///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool search_volume_exit_fragment ( __global uint4*  indexlist,
                                   unsigned         volume_id, 
                                   unsigned         fragindex, 
                                   int              remaining_fragments,
                                   uint4*           exit_fragment,
                                   unsigned*        nfragments_found)
{
  bool exit_found = false;

  for (int i = 0; i < remaining_fragments; ++i)
  {
    uint4 fraginfo = indexlist[fragindex];
    fragindex      = fraginfo.x;

    if ( fraginfo.y == volume_id )
    {
      exit_found     = true;
      *exit_fragment = fraginfo;
      ++(*nfragments_found);
    }
  }

  return exit_found;
}

#endif

