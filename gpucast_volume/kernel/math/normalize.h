#ifndef LIBGPUCAST_NORMALIZE_H
#define LIBGPUCAST_NORMALIZE_H

#include <math/length.h>
#include <math/operator.h>

// absolut normalize
template <typename T>
__device__
inline T normalize ( T const& v )
{
  return v / length(v);
}

// relative normalize
template <typename T>
__device__ inline
T normalize ( T const& v, T const& vmin, T const& vmax )
{
  return ( v - vmin ) / ( vmax - vmin );
}

#endif