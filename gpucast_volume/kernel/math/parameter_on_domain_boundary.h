#ifndef LIB_GPUCAST_PARAMETER_ON_DOMAIN_BOUNDARY_H
#define LIB_GPUCAST_PARAMETER_ON_DOMAIN_BOUNDARY_H

///////////////////////////////////////////////////////////////////////////////
__device__ inline
bool parameter_on_domain_boundary ( float3 const& uvw, 
                                    float3 const& uvwmin_local,
                                    float3 const& uvwmax_local,
                                    float3 const& uvwmin_global,
                                    float3 const& uvwmax_global,
                                    float  epsilon )
{
  float3 uvw_global = uvwmin_local + uvw * (uvwmax_local - uvwmin_local);

  return uvw_global.x > uvwmax_global.x - epsilon ||
         uvw_global.y > uvwmax_global.y - epsilon ||
         uvw_global.z > uvwmax_global.z - epsilon ||
         uvw_global.x < uvwmin_global.x + epsilon ||
         uvw_global.y < uvwmin_global.y + epsilon ||
         uvw_global.z < uvwmin_global.z + epsilon;
}

#endif // LIB_GPUCAST_PARAMETER_ON_DOMAIN_BOUNDARY_H
