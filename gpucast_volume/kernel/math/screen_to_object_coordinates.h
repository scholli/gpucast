#ifndef LIBGPUCAST_SCREEN_TO_OBJECT_COORDINATES_H
#define LIBGPUCAST_SCREEN_TO_OBJECT_COORDINATES_H

#include <math/mult.h>

__device__ inline void 
screen_to_object_coordinates ( int2 screen_coords, 
                               int2 screen_resolution, 
                               float depth, 
                               float4 const* modelviewprojectionmatrixinverse,
                               float4* object_coordinates )
{
  float4  screenposition     = float4_t ( (float(screen_coords.x) + 0.5) / float(screen_resolution.x) * 2.0f - 1.0f, 
                                          (float(screen_coords.y) + 0.5) / float(screen_resolution.y) * 2.0f - 1.0f, 
                                          depth * 2.0 - 1.0f, 
                                          1.0f );
  *object_coordinates        = mult_mat4_float4 ( modelviewprojectionmatrixinverse, screenposition );
  *object_coordinates        = (*object_coordinates) / (*object_coordinates).w;
}

__device__ inline float4 
screen_to_object_coordinates ( int2 screen_coords, 
                               int2 screen_resolution, 
                               float depth, 
                               float4 const* modelviewprojectionmatrixinverse )
{
  // recalculate screen coordinates
  float4  screenposition     = float4_t ( (float(screen_coords.x) + 0.5) / float(screen_resolution.x) * 2.0f - 1.0f, 
                                          (float(screen_coords.y) + 0.5) / float(screen_resolution.y) * 2.0f - 1.0f, 
                                          depth * 2.0 - 1.0f, 
                                          1.0f );

  // transform to homogenous objectspace and project to euclidian coordinates
  float4 object_coordinates  = mult_mat4_float4 ( modelviewprojectionmatrixinverse, screenposition );
  object_coordinates         = object_coordinates / object_coordinates.w;

  return object_coordinates;
}



#endif // LIBGPUCAST_SCREEN_TO_OBJECT_COORDINATES_H