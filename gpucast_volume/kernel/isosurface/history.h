#ifndef LIBGPUCAST_HISTORY_H
#define LIBGPUCAST_HISTORY_H

///////////////////////////////////////////////////////////////////////////////
struct history
{
  __device__ inline history( unsigned* shared_mem )
    : data (shared_mem),
      size (0)
  {}

  __device__ inline void
  push_back (unsigned k) 
  {
    data[size] = k;
    ++size;
  }

  __device__ inline bool 
  contains (unsigned volume_id) const
  {
    for (int i = 0; i != size; ++i ) {
      if ( data[i] == volume_id ) {
        return true;
      }
    }

    return false;
  }

  unsigned  size;
  unsigned* data;
};

#endif // LIBGPUCAST_HISTORY_H