#ifndef LIB_GPUCAST_IN_DOMAIN_H
#define LIB_GPUCAST_IN_DOMAIN_H

/*********************************************************************
 * checks if value is in domain
 *********************************************************************/
__device__
inline bool
in_domain ( float v, float vmin, float vmax )
{
  return v >= vmin && v <= vmax;
}

__device__
inline bool
in_domain ( float2 const& v, float2 const& vmin, float2 const& vmax )
{
  return v.x >= vmin.x && v.x <= vmax.x &&
         v.y >= vmin.y && v.y <= vmax.y;
}

__device__
inline bool
in_domain ( float3 const& v, float3 const& vmin, float3 const& vmax )
{
  return v.x >= vmin.x && v.x <= vmax.x &&
         v.y >= vmin.y && v.y <= vmax.y &&
         v.z >= vmin.z && v.z <= vmax.z;
}

__device__
inline bool
in_domain ( float4 const& v, float4 const& vmin, float4 const& vmax )
{
  return v.x >= vmin.x && v.x <= vmax.x &&
         v.y >= vmin.y && v.y <= vmax.y &&
         v.z >= vmin.z && v.z <= vmax.z &&
         v.w >= vmin.w && v.w <= vmax.w;
}

#endif // LIB_GPUCAST_IN_DOMAIN_H