#ifndef LIBGPUCAST_CUDA_SWAP_H
#define LIBGPUCAST_CUDA_SWAP_H

template <typename T>
__device__
inline void swap ( T& lhs, T& rhs ) 
{
  T tmp(lhs);
  lhs = rhs;
  rhs = tmp;
}

#endif // LIBGPUCAST_CUDA_SWAP_H