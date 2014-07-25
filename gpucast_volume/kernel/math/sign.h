#ifndef LIBGPUCAST_CUDA_SIGN_H
#define LIBGPUCAST_CUDA_SIGN_H

__device__
inline float sign ( float x ) 
{
  return x < 0.0f ? -1.0f : 1.0f;
}
#endif // LIBGPUCAST_CUDA_SIGN_H