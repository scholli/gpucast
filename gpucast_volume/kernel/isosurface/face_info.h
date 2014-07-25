#ifndef LIBGPUCAST_FACE_INFO_H
#define LIBGPUCAST_FACE_INFO_H

#include <math/conversion.h>
#include <math/cross.h>
#include <math/normalize.h>
#include <math/mult.h>
#include <math/horner_surface.h>
#include <math/transferfunction.h>

#include <shade/faceforward.h>

#include <isosurface/fragment.h>

///////////////////////////////////////////////////////////////////////////////
struct face_info
{
  __device__ inline face_info()
    : volume_data_id          (0),
      adjacent_volume_data_id (0),
      surface_unique_id       (0),
      surface_points_id       (0),
      is_surface              (false),
      is_outer                (false),
      depth                   (0.0f),
      order                   ()
  {}

  __device__ inline face_info( uint4 const* surfacedatabuffer, 
                               float4 const* volumedatabuffer, 
                               unsigned surface_data_id )
    : volume_data_id          (0),
      adjacent_volume_data_id (0),
      surface_unique_id       (0),
      surface_points_id       (0),
      is_surface              (false),
      is_outer                (false),
      depth                   (0.0f),
      order                   ()
  {
    surface_unique_id       = surfacedatabuffer[surface_data_id    ].x;
    volume_data_id          = surfacedatabuffer[surface_data_id    ].z;

    surface_points_id       = surfacedatabuffer[surface_data_id + 1].y;
    adjacent_volume_data_id = surfacedatabuffer[surface_data_id + 1].z;

    is_outer                = surfacedatabuffer[surface_data_id + 2].x;
    order                   = uint2_t ( surfacedatabuffer[surface_data_id + 2].y,
                                        surfacedatabuffer[surface_data_id + 2].z );
  }
  __device__ inline face_info( uint4 const* surfacedatabuffer, 
                               float4 const* volumedatabuffer, 
                               fragment const& f )
    : volume_data_id          (0),
      adjacent_volume_data_id (0),
      surface_unique_id       (0),
      surface_points_id       (0),
      is_surface              (false),
      is_outer                (false),
      depth                   (0.0f),
      order                   ()
  {
    surface_unique_id       = surfacedatabuffer[f.surface_data_id    ].x;
    volume_data_id          = surfacedatabuffer[f.surface_data_id    ].z;

    surface_points_id       = surfacedatabuffer[f.surface_data_id + 1].y;
    adjacent_volume_data_id = surfacedatabuffer[f.surface_data_id + 1].z;

    is_surface              = f.uv.x > -0.5f;
    depth                   = f.depth;

    is_outer                = surfacedatabuffer[f.surface_data_id + 2].x;
    order                   = uint2_t ( surfacedatabuffer[f.surface_data_id + 2].y,
                                        surfacedatabuffer[f.surface_data_id + 2].z );
  }

  __device__ inline float 
  shade  ( float2 const&  uv,
           float4 const*  surfacepointsbuffer,
           float4 const*  modelviewmatrix,
           float4 const*  normalmatrix ) const
  {
    float4 intersection0, du, dv;
    horner_surface_derivatives ( surfacepointsbuffer, surface_points_id, order, uv, intersection0, du, dv);
    float3  intersection_normal = normalize ( cross ( float3_t(du.x, du.y, du.z), float3_t(dv.x, dv.y, dv.z) ) );
        
    float4 surf_normal = float4_t(intersection_normal.x, intersection_normal.y, intersection_normal.z, 0.0f);
    float4 surf_point  = float4_t(intersection0.x, intersection0.y, intersection0.z, 1.0f);
    float4 lightpos    = float4_t(0.0f, 0.0f, 0.0f, 1.0f); // light from camera
    float4 pworld      = mult_mat4_float4 ( modelviewmatrix, surf_point );
    surf_normal        = mult_mat4_float4 ( normalmatrix, surf_normal ); 

    float3 L           = normalize ( float3_t(lightpos.x, lightpos.y, lightpos.z) - float3_t(pworld.x, pworld.y, pworld.z) );
    float3 N           = normalize ( float3_t(surf_normal.x, surf_normal.y, surf_normal.z) );
    N                  = faceforward ( float3_t(-pworld.x, -pworld.y, -pworld.z), N );
    float diffuse      = dot (N , L);
    return diffuse;
  }

  unsigned  volume_data_id;
  unsigned  adjacent_volume_data_id;
  unsigned  surface_unique_id;
  unsigned  surface_points_id;

  uint2     order;

  bool      is_surface;
  bool      is_outer;

  float     depth;
};

#endif // LIBGPUCAST_FACE_INFO_H