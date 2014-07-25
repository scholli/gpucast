#ifndef LIBGPUCAST_SAMPLE_H
#define LIBGPUCAST_SAMPLE_H

#include <math/in_domain.h>
#include <shade/faceforward.h>

#include <isosurface/volume_info.h>
#include <isosurface/compute_iso_normal.h>

struct isoparametric_transition
{
  float2 uv[6];   // umin=0, umax=1, vmin=2, vmax=3, wmin=4, wmax=5
  bool   hit[6];  // umin=0, umax=1, vmin=2, vmax=3, wmin=4, wmax=5

  __device__ inline bool
  intersects_boundary () const
  {
    return hit[0] || hit[1] || 
           hit[2] || hit[3] || 
           hit[4] || hit[5];
  }
};

///////////////////////////////////////////////////////////////////////////////
struct sample
{

  ///////////////////////////////////////////////////////////////////////////////
  __device__ sample ()
    : inversion_success (false),
      is_inside         (false)
  {}

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool 
  is_in_volume () const 
  { 
    return volume.volume_data_id != 0; 
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float4
  compute_isosurface_normal () const
  {
    return compute_iso_normal (dp_du, dp_dv, dp_dw, da_du, da_dv, da_dw);
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float4
  compute_face_normal ( unsigned face_type ) const
  {
    if ( face_type == 0 || face_type == 1 ) return normalize ( cross ( dp_dv, dp_dw ) );
    if ( face_type == 2 || face_type == 3 ) return normalize ( cross ( dp_du, dp_dw ) );
    if ( face_type == 4 || face_type == 5 ) return normalize ( cross ( dp_du, dp_dv ) );

    return float4_t(0.0, 0.0, 0.0, 0.0);
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline bool 
  uvw_in_domain () const
  {
    return in_domain(uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline unsigned 
  compute_face_type ( sample const& other ) const 
  {
    if ( (uvw.x  ) * (other.uvw.x  ) < 0.0f ) return 0;
    if ( (uvw.x-1) * (other.uvw.x-1) < 0.0f ) return 1;
    if ( (uvw.y  ) * (other.uvw.y  ) < 0.0f ) return 2;
    if ( (uvw.y-1) * (other.uvw.y-1) < 0.0f ) return 3;
    if ( (uvw.z  ) * (other.uvw.z  ) < 0.0f ) return 4;
    if ( (uvw.z-1) * (other.uvw.z-1) < 0.0f ) return 5;
    return 6;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline void 
  compute_transition ( sample const& other, isoparametric_transition& transition ) const 
  {
    transition.hit[0] = (uvw.x  ) * (other.uvw.x  ) < 0.0f;
    transition.hit[1] = (uvw.x-1) * (other.uvw.x-1) < 0.0f;
    transition.hit[2] = (uvw.y  ) * (other.uvw.y  ) < 0.0f;
    transition.hit[3] = (uvw.y-1) * (other.uvw.y-1) < 0.0f;
    transition.hit[4] = (uvw.z  ) * (other.uvw.z  ) < 0.0f;
    transition.hit[5] = (uvw.z-1) * (other.uvw.z-1) < 0.0f;

    float3 uvw_interp = (uvw + other.uvw) / 2.0f;
    transition.uv[0]  = float2_t(uvw_interp.y, uvw_interp.z);
    transition.uv[1]  = float2_t(uvw_interp.y, uvw_interp.z);
    transition.uv[2]  = float2_t(uvw_interp.x, uvw_interp.z);
    transition.uv[3]  = float2_t(uvw_interp.x, uvw_interp.z);
    transition.uv[4]  = float2_t(uvw_interp.x, uvw_interp.y);
    transition.uv[5]  = float2_t(uvw_interp.x, uvw_interp.y);
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float4
  shade_isosurface ( float4 const* modelviewmatrix,
                     float4 const* normalmatrix,
                     float global_attribute_min,
                     float global_attribute_max ) const
  {
    return shade ( modelviewmatrix, normalmatrix, compute_isosurface_normal(), global_attribute_min, global_attribute_max);
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float4
  shade ( float4 const* modelviewmatrix,
          float4 const* normalmatrix,
          float4 const& normal,
          float         global_attribute_min,
          float         global_attribute_max ) const
  {
    float4 lightpos       = float4_t(0.0f, 0.0f, 0.0f, 1.0f);
    float4 pworld         = modelviewmatrix * p;
    float4 nworld         = modelviewmatrix * normal; 
                          
    float3 L              = normalize ( float3_t(lightpos.x, lightpos.y, lightpos.z) - float3_t(pworld.x, pworld.y, pworld.z) );
    float3 N              = normalize ( float3_t(nworld.x, nworld.y, nworld.z) );

    N                     = faceforward ( float3_t(-pworld.x, -pworld.y, -pworld.z), N );
    float diffuse         = dot (N , L);

    float relative_attrib = normalize ( a.x, global_attribute_min, global_attribute_max );
    float4 relative_color = diffuse * transferfunction ( relative_attrib );
    //float4 relative_color = float4_t(N.x, N.y, N.z, 1.0);
    relative_color.w      = 1.0f;

    return relative_color;
  }

  ///////////////////////////////////////////////////////////////////////////////
  __device__ inline float4
  shade_face ( float4 const* modelviewmatrix,
               float4 const* normalmatrix,
               unsigned face_type,
               float global_attribute_min,
               float global_attribute_max ) const
  {
    return shade ( modelviewmatrix, normalmatrix, compute_face_normal(face_type), global_attribute_min, global_attribute_max);
  }

  // attributes 
  volume_info volume;

  bool        inversion_success;
  bool        is_inside;

  float3      uvw;

  float4      p;

  float4      dp_du;
  float4      dp_dv;
  float4      dp_dw;

  float2      a;

  float2      da_du;
  float2      da_dv;
  float2      da_dw;
};

#endif // LIBGPUCAST_TARGET_FUNCTION_H