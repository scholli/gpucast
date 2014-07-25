#ifndef LIBGPUCAST_BUBBLE_SORT_INDEXLIST_H
#define LIBGPUCAST_BUBBLE_SORT_INDEXLIST_H

#include "math/conversion.h"


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void bubble_sort_indexlist_1 ( int             start_index,
                               unsigned        nfragments,
                               __global uint4*  indexlist )
{
  // sort list of fragments
  for ( int i = 0; i != nfragments; ++i )
  { 
    int index   = start_index;

    for ( int j = 0; j != int(nfragments-1); ++j )
    { 
      // get entry for this fragment
      uint4 entry0 = indexlist[index];

      if ( entry0.x == 0 ) {
        break;
      } else {
        uint4 entry1 = indexlist[int_t(entry0.x)];

        if ( intBitsToFloat(int_t(entry0.w)) > intBitsToFloat(int_t(entry1.w)) ) 
        {
          // swizzle depth and related data
          indexlist[index          ] = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
          indexlist[int_t(entry0.x)] = uint4_t(entry1.x, entry0.y, entry0.z, entry0.w);
        }

        index = int_t(entry0.x);
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void bubble_sort_indexlist_by_id ( int      start_index,
                                   unsigned nfragments,
                                   uint4*   indexlist )
{
  // sort list of fragments
  for ( int i = 0; i != nfragments; ++i )
  { 
    int index   = start_index;

    for ( int j = 0; j != int(nfragments-1); ++j )
    { 
      // get entry for this fragment
      uint4 entry0 = indexlist[index];

      if ( entry0.x == 0 ) {
        break;
      } else {
        uint4 entry1 = indexlist[int_t(entry0.x)];

        if ( entry0.z > entry1.z ) 
        {
          // swizzle depth and related data
          indexlist[index          ] = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
          indexlist[int_t(entry0.x)] = uint4_t(entry1.x, entry0.y, entry0.z, entry0.w);
        }

        index = int_t(entry0.x);
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool get_next_fragment ( uint4 const& fragment,
                         uint4 const* indexlist,
                         int&         next_index,
                         uint4&       next_fragment ) 
{
  next_fragment = fragment;
  while ( next_fragment.x != 0 )
  {
    next_index    = next_fragment.x;
    next_fragment = indexlist[next_index];

    if ( next_fragment.z == fragment.z )
    {
      return true;
    } 
  }

  return false;
}
                         
///////////////////////////////////////////////////////////////////////////////
__device__ inline
void shift_fragments ( int    first_index,
                       int    source_index,
                       uint4* indexlist )
{
  uint4 source             = indexlist[source_index];

  int   backup_index       = first_index;
  uint4 backup_fragment    = indexlist[backup_index];

  indexlist[first_index]   = uint4_t(backup_fragment.x, source.y, source.z, source.w);

  while ( backup_fragment.x != source_index ) 
  {
    uint4 tmp = indexlist[backup_fragment.x];
    indexlist[backup_fragment.x] = uint4_t(tmp.x, backup_fragment.y, backup_fragment.z, backup_fragment.w);
    backup_fragment = tmp;
  }

  indexlist[source_index] = uint4_t(source.x, backup_fragment.y, backup_fragment.z, backup_fragment.w);
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void merge_intervals ( int      start_index,
                       uint4*   indexlist )
{
  int   index    = start_index;
  uint4 fragment = indexlist[index];

  int   next_index = 0;
  uint4 next_fragment;

  while ( true )
  {

    if ( !get_next_fragment ( fragment, indexlist, next_index, next_fragment )) {
      break;
    }
  
    if ( next_index != fragment.x )
    {
      shift_fragments(fragment.x, next_index, indexlist);
    } 

    next_fragment = indexlist[fragment.x];

    if ( next_fragment.x != 0 )
    {
      index    = next_fragment.x;
      fragment = indexlist[index];
    } else {
      break;
    }
  }
}



///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool verify_indexlist_for_domain_intersection ( int      start_index,
                                                uint4*   indexlist )
{
  uint4    entry      = indexlist[start_index];
  unsigned transition = 0;

  while ( entry.x != 0 )
  {
    uint4 tmp = entry;
    entry     = indexlist[entry.x];

    if ( tmp.z == entry.z )
    {
      transition = 0;
      if ( intBitsToFloat(int_t(tmp.w)) > intBitsToFloat(int_t(entry.w)) ) 
      {
        return false;
      }
    } else {
      ++transition;
    }

    if ( transition > 1 )
    {
      return false;
    }
  }
  return true;
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void sort_fragments_per_chull ( int      start_index,
                                unsigned nfragments,
                                uint4*   indexlist )
{
  int index       = start_index;

  while ( index != 0 )
  {
    uint4 entry0 = indexlist[index];

    if ( entry0.x == 0 )
    {
      index = 0;
    } else {

      uint4 entry1 = indexlist[int_t(entry0.x)];

      if ( intBitsToFloat(int_t(entry0.w)) > intBitsToFloat(int_t(entry1.w)) &&
           entry0.z == entry1.z ) 
      {
        // swizzle depth and related data
        indexlist[index          ] = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
        indexlist[int_t(entry0.x)] = uint4_t(entry1.x, entry0.y, entry0.z, entry0.w);
      }

      index = entry1.x;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void bubble_sort_indexlist_2 ( int             start_index,
                               unsigned        nfragments,
                               __global uint4*  indexlist )
{
  // sort list of fragments
  for ( int i = 0; i != nfragments; ++i )
  { 
    int   index0 = start_index;
    uint4 entry0 = indexlist[index0];
    int   index1 = entry0.x;

    while ( index1 != 0 )
    {
      uint4 entry1 = indexlist[index1];

      if ( intBitsToFloat(entry0.w) > intBitsToFloat(entry1.w) ) 
      {
        // swizzle depth and related data
        indexlist[index0] = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
        indexlist[index1] = uint4_t(entry1.x, entry0.y, entry0.z, entry0.w);
        entry0 = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
      } else {
        entry0 = entry1;
      }

      index1 = entry1.x;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void bubble_sort_indexlist_3 ( int      start_index,
                               unsigned nfragments,
                               uint4*   indexlist )
{
  // sort list of fragments
  for ( int i = 0; i != nfragments; ++i )
  {
    int   index0 = start_index;

    for ( int j = 0; j != int(nfragments-1); ++j )
    {
      uint4 entry0 = indexlist[index0];
      int   index1 = entry0.x;
      uint4 entry1 = indexlist[index1];

      if ( intBitsToFloat(entry0.w) > intBitsToFloat(entry1.w) ) 
      {
        // swizzle depth and related data
        indexlist[index0] = uint4_t(entry0.x, entry1.y, entry1.z, entry1.w); 
        indexlist[index1] = uint4_t(entry1.x, entry0.y, entry0.z, entry0.w);
      }

      index0 = index1;
    }
  }
}

#endif // LIBGPUCAST_BUBBLE_SORT_INDEXLIST_H

