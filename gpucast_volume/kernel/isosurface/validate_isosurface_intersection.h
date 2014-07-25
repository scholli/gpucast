#ifndef LIBGPUCAST_VALIDATE_ISOSURFACE_INTERSECTION_H
#define LIBGPUCAST_VALIDATE_ISOSURFACE_INTERSECTION_H

#include ".\local_memory_config.h"

#include "isosurface/validate_bezier_domain.h"
#include "isosurface/target_function.h"
#include "math/point_ray_distance.h"
#include "math/horner_volume.h"

/////////////////////////////////////////////////////////////////////////////
__device__ inline
bool validate_isosurface_intersection ( float4 const* volumepointsbuffer,
                                        float2 const* attributepointsbuffer,
                                        unsigned      point_baseid,
                                        unsigned      attribute_baseid,
                                        uint3  const& order,
                                        float4 const& ray_entry,
                                        float4 const& ray_direction,
                                        float3 const& uvw,
                                        float         iso_threshold,
                                        float         intersection_epsilon )
{
  float4 point         = horner_volume<float4, 3> ( volumepointsbuffer, point_baseid, order, uvw );
  //float4 attrib        = horner_volume ( attributebuffer, attribute_baseid, order, uvw );
  //float relative_error = fabs( target_function(attrib) - target_function(iso_threshold) ) / fabs( target_function( attrib ));

  return //relative_error < intersection_epsilon &&
         in_domain3(uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f) )  && 
         point_ray_distance(float3_t(ray_entry.x, ray_entry.y, ray_entry.z), 
                            float3_t(ray_direction.x, ray_direction.y, ray_direction.z), 
                            float3_t(point.x, point.y, point.z)) < intersection_epsilon;
}

#endif // LIBGPUCAST_VALIDATE_ISOSURFACE_INTERSECTION_H