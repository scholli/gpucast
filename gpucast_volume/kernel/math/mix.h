#ifndef LIBGPUCAST_CUDA_MIX_H
#define LIBGPUCAST_CUDA_MIX_H

#include <math/operator.h>

template <typename T>
__device__
inline T mix ( T const lhs, T const& rhs, float a ) 
{
  return (1.0f - a ) * lhs + a * rhs;
}

#endif // LIBGPUCAST_CUDA_MIX_H