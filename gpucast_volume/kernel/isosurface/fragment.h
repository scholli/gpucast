#ifndef LIBGPUCAST_FRAGMENT_H
#define LIBGPUCAST_FRAGMENT_H

#include <isosurface/ray_state.h>


__device__ inline float3 compute_uvw_from_uv ( uint4 const*   surfacedatabuffer, 
                                               unsigned       surface_data_id,
                                               float2 const&  uv)
{
  // get swizzle info
  uint4   surface_param_info          = surfacedatabuffer [surface_data_id + 3];
  uint3   uvw_swizzle                 = uint3_t ( surface_param_info.x, surface_param_info.y, surface_param_info.z);
  float   w                           = clamp ( float(surface_param_info.w), 0.0f, 1.0f );
  float   uvw[3];
  uvw[uvw_swizzle.x]                  = uv.x;
  uvw[uvw_swizzle.y]                  = uv.y;
  uvw[uvw_swizzle.z]                  = w;
  float3  start_uvw                   = float3_t( uvw[0], uvw[1], uvw[2] ); 
  start_uvw                           = clamp ( start_uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f) );
  return start_uvw;
}


///////////////////////////////////////////////////////////////////////////////
struct fragment
{
  ///////////////////////////////////////////////////////////////////////////////
  unsigned     next;
  float        depth;
  unsigned     surface_data_id;
  float2       uv;

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline 
  fragment()
    : surface_data_id (0)
  {}

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline 
  fragment ( unsigned index, uint4 const* buffer )
  {
    uint4 const data = buffer[index];
    
    next             = data.x;
    depth            = intBitsToFloat(data.w);
    uv               = unpack_uv (data.y);
    surface_data_id  = data.z;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool 
  has_next () const
  {
    return next != 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool 
  is_valid () const
  {
    return surface_data_id != 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void 
  swap ( fragment& other )
  {
    fragment tmp = other;
    other = *this;
    *this = tmp;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float3 
  compute_uvw ( uint4 const* surfacedatabuffer, 
                float2 const& uv) const
  {
    return compute_uvw_from_uv ( surfacedatabuffer, surface_data_id, uv );
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float3 
  uvw ( uint4 const* surfacedatabuffer,
        unsigned     volume_data_id ) const
  {
    return uvw ( surfacedatabuffer, volume_data_id, uv );
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float3 
  uvw ( uint4 const*  surfacedatabuffer,
        unsigned      volume_data_id,
        float2 const& uv_guess ) const
  {
    float3 uvw = compute_uvw (surfacedatabuffer, uv_guess);

    if ( surfacedatabuffer[surface_data_id].z == volume_data_id )
    {
      return uvw;
    } else { // use adjacent volume -> transform uvw to adjacent volume 
      unsigned surface_type = surfacedatabuffer[surface_data_id+2].w;
      float3 offset         = float3_t(0.0f, 0.0f, 0.0f);
      offset.x += float ( surface_type == 0 ); // u=0 becomes u=1 for adjacent volume
      offset.y += float ( surface_type == 2 ); // v=0 becomes v=1 for adjacent volume
      offset.z += float ( surface_type == 4 ); // w=0 becomes w=1 for adjacent volume
      return uvw + offset;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void 
  get_volume_ids ( uint4 const* surfacedatabuffer, unsigned& id0, unsigned& id1 ) const
  {
    id0 = surfacedatabuffer[surface_data_id  ].z;
    id1 = surfacedatabuffer[surface_data_id+1].z;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline fragment 
  get_next ( uint4 const* buffer ) const
  {
    return fragment ( next, buffer );
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool
  belongs_to_outer_surface ( uint4 const* surfacedatabuffer ) const
  {
    return surfacedatabuffer[surface_data_id+2].x != 0;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool
  intersect_outer_surface ( uint4 const*     surfacedatabuffer, 
                            float4 const*    surfacepointsbuffer, 
                            ray_state const& ray,
                            float            fixed_newton_epsilon,
                            float            max_newton_steps,
                            float2&          uv_face,
                            float4&          p,
                            float4&          du,
                            float4&          dv )
  {
    unsigned surface_points_id = surfacedatabuffer[surface_data_id+1].y;
    uint4 data    = surfacedatabuffer[surface_data_id+2];

    bool is_outer = data.x != 0;
    uint2 order   = uint2_t(data.y, data.z);

    if ( !is_outer ) 
    {
      return false; // is inner surface -> not relevant for surface rendering
    } else {
      unsigned iterations;

      // start at guess
      uv_face = uv;

      return newton_surface ( surfacepointsbuffer, 
                              surface_points_id, 
                              uv_face, 
                              fixed_newton_epsilon, 
                              max_newton_steps, 
                              order,
                              ray.n1, 
                              ray.n2,
                              ray.d1,
                              ray.d2,
                              p,
                              du,
                              dv,
                              iterations );
    }
  }
    
};

#endif // LIBGPUCAST_TARGET_FUNCTION_H