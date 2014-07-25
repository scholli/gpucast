#ifndef LIB_GPUCAST_FACEFORWARD_H
#define LIB_GPUCAST_FACEFORWARD_H

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float3 faceforward ( float3 const& to_camera, float3 const& normal )
{
  return ( dot (to_camera, normal) < 0.0) ? float3_t(-normal.x, -normal.y, -normal.z) : normal;
}

#endif // LIB_GPUCAST_FACEFORWARD_H
