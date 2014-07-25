#ifndef LIBGPUCAST_FRAGMENT_RANGE_H
#define LIBGPUCAST_FRAGMENT_RANGE_H

#include <math/conversion.h>



///////////////////////////////////////////////////////////////////////////////
class fragment_range
{
private :
  ///////////////////////////////////////////////////////////////////////////////
  fragment*           fragments;      // memory for hits is allocated outside
  unsigned            nfragments;     // number of convex hull fragments for this volume
  unsigned            volume_data_id; // volume index
  bool                abort_range;    // stop processing this range  

public :

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline
  fragment_range ( fragment* shared_mem_fragments )
  : fragments      (shared_mem_fragments),
    nfragments     (0), 
    volume_data_id (0),
    abort_range    (false)
  {}

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool 
  valid () const
  {
    return volume_data_id != 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline unsigned 
  size() const
  {
    return nfragments;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline unsigned const&
  get_volume_id () const
  {
    return volume_data_id;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void 
  initialize ( fragment const& f, unsigned const& vid )
  {
    fragments[nfragments] = f;
    ++nfragments;
    volume_data_id = vid;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void 
  add ( fragment const& f )
  {
    fragments[nfragments] = f;
    ++nfragments;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void 
  clear ()
  {
    nfragments     = 0;
    volume_data_id = 0;
    abort_range    = false;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool
  abort () const
  {
    return abort_range;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void
  abort ( bool a )
  {
    abort_range = a;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void
  sort ()
  {
    for (int n=nfragments; n>1; n=n-1) {
      for (int i=0; i<n-1; i=i+1) {
        if (fragments[i].depth > fragments[i+1].depth) {
          fragments[i].swap(fragments[i+1]);
        } 
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool
  is_sorted ()
  {
    for (int i = 0; i < int(nfragments)-1; ++i) {
      if (fragments[i].depth > fragments[i+1].depth) {
        return false;
      }
    }
    return true;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline unsigned 
  get_surface_data_id ( uint4 const* surfacedatabuffer, 
                        float depth,
                        unsigned face_type ) const
  {
    for ( int i = 0; i < nfragments; ++i ) 
    {
      if ( fragments[i].depth >= depth &&
           surfacedatabuffer[fragments[i].surface_data_id+2].w == face_type )
      {
        return fragments[i].surface_data_id;
      }
    }
    return 0; // no face intersection
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline fragment const&
  first_fragment () const
  {
    return fragments[0];
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline fragment const&
  last_fragment () const
  {
    return fragments[nfragments-1];
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float3 
  interpolate_guess ( float guess_depth, uint4 const* surfacedatabuffer ) 
  {
    for ( int i = 0; i < int(nfragments)-1; ++i ) 
    {
      if ( fragments[i  ].depth >= guess_depth && 
           fragments[i+1].depth <= guess_depth )
      {
        float dist_total = fragments[i+1].depth - fragments[i].depth;
        float dist_guess = guess_depth - fragments[i].depth;
        return mix ( fragments[i].uvw(surfacedatabuffer, volume_data_id), 
                     fragments[i+1].uvw(surfacedatabuffer, volume_data_id), 
                     dist_guess/dist_total );
      }
    }
    // fallback -> should not happen
    return float3_t(0.0f, 0.0f, 0.0f);
  }

};

#endif // LIBGPUCAST_FRAGMENT_RANGE_H