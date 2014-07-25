#ifndef LIB_GPUCAST_COMPUTE_DEPTH_H
#define LIB_GPUCAST_COMPUTE_DEPTH_H

#include "math/mult.h"
#include "math/clamp.h"

///////////////////////////////////////////////////////////////////////////////
__device__  inline
  float compute_depth_from_world_coordinates ( float4 point_worldcoordinates,
                                               float  nearplane,
                                               float  farplane ) 
{
  float depth = (point_worldcoordinates.w/point_worldcoordinates.z) * farplane * (nearplane / (farplane - nearplane)) + 0.5f * (farplane + nearplane)/(farplane - nearplane) + 0.5f;
  return clamp (depth, 0.0f, 1.0f);
}


///////////////////////////////////////////////////////////////////////////////
__device__  inline
  float compute_depth_from_object_coordinates ( float4 const* modelviewmatrix, 
                                                float4        point_objectcoordinates, 
                                                float         nearplane,
                                                float         farplane )
{
  point_objectcoordinates.w = 1.0;
  return compute_depth_from_world_coordinates ( modelviewmatrix * point_objectcoordinates, nearplane, farplane );  
}

#endif // LIB_GPUCAST_COMPUTE_DEPTH_H
