#ifndef LIBGPUCAST_VALIDATE_BEZIER_DOMAIN_H
#define LIBGPUCAST_VALIDATE_BEZIER_DOMAIN_H

/////////////////////////////////////////////////////////////////////////////
__device__ inline
bool in_domain2 ( float2 const& uvw, float2 const& min, float2 const& max  )
{
  return  uvw.x >= min.x && 
          uvw.y >= min.y && 
          uvw.x <= max.x && 
          uvw.y <= max.y; 
}

/////////////////////////////////////////////////////////////////////////////
__device__ inline
bool in_domain3 ( float3 const& uvw, float3 const& min, float3 const& max )
{
  return  uvw.x >= min.x && 
          uvw.y >= min.y && 
          uvw.z >= min.z && 
          uvw.x <= max.x && 
          uvw.y <= max.y && 
          uvw.z <= max.z; 
}

#endif // LIBGPUCAST_VALIDATE_BEZIER_DOMAIN_H