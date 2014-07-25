#ifndef LIBGPUCAST_COMPUTE_ISO_NORMAL_H
#define LIBGPUCAST_COMPUTE_ISO_NORMAL_H

#include "isosurface/target_function.h"
#include "math/mult.h"
#include "math/cross.h"

/////////////////////////////////////////////////////////////////////////////
__device__  inline
float4 compute_iso_normal ( float4 const& du, 
                            float4 const& dv, 
                            float4 const& dw, 
                            float2 const& ddu, 
                            float2 const& ddv, 
                            float2 const& ddw)
{
  // partial derivative dD / duvw
  float3 dD_duvw = normalize( float3_t ( target_function(ddu.x), 
                                         target_function(ddv.x),
                                         target_function(ddw.x) ) );

  // solutions for 0 = dD / duvw * duvw0
  float3 duvw0; 
  float3 duvw1;

  // two solutions of overdetermined attribute root-function
#if 0
  if ( fabs(dD_duvw.z) > 0.0f && fabs(dD_duvw.y) > 0.0f )
  {
    duvw0 = normalize ( float3_t ( 1.0f, 1.0f, (-dD_duvw.x-dD_duvw.y)/dD_duvw.z ) );
    duvw1 = normalize ( float3_t ( 1.0f, (-dD_duvw.x-dD_duvw.z)/dD_duvw.y, 1.0f ) );
  } else if ( fabs(dD_duvw.x) > 0.0f && fabs(dD_duvw.y) > 0.0f ) { 
    duvw0 = normalize ( float3_t ( (-dD_duvw.y-dD_duvw.z)/dD_duvw.x, 1.0f, 1.0f ) );
    duvw1 = normalize ( float3_t ( 1.0f, (-dD_duvw.x-dD_duvw.z)/dD_duvw.y, 1.0f ) );
  } else if ( fabs(dD_duvw.x) > 0.0f && fabs(dD_duvw.z) > 0.0f ) {
    duvw0 = normalize ( float3_t ( (-dD_duvw.y-dD_duvw.z)/dD_duvw.x, 1.0f, 1.0f ) );
    duvw1 = normalize ( float3_t ( 1.0f, 1.0f, (-dD_duvw.x-dD_duvw.y)/dD_duvw.z ) );
  }

  float3 M[3]  = { normalize(float3_t (du.x, du.y, du.z)), 
                  normalize(float3_t (dv.x, dv.y, dv.z)), 
                  normalize(float3_t (dw.x, dw.y, dw.z)) };
#else
  if ( fabs(dD_duvw.z) > 0.0f && fabs(dD_duvw.y) > 0.0f )
  {
    duvw0 = normalize ( float3_t ( 1.0f, 1.0f, (-dD_duvw.x-dD_duvw.y)/dD_duvw.z ) );
    duvw1 = normalize ( float3_t ( 1.0f, (-dD_duvw.x-dD_duvw.z)/dD_duvw.y, 1.0f ) );
  } else if ( fabs(dD_duvw.x) > 0.0f && fabs(dD_duvw.y) > 0.0f ) { 
    duvw0 = normalize ( float3_t ( (-dD_duvw.y-dD_duvw.z)/dD_duvw.x, 1.0f, 1.0f ) );
    duvw1 = normalize ( float3_t ( 1.0f, (-dD_duvw.x-dD_duvw.z)/dD_duvw.y, 1.0f ) );
  } else if ( fabs(dD_duvw.x) > 0.0f && fabs(dD_duvw.z) > 0.0f ) {
    duvw0 = normalize ( float3_t ( (-dD_duvw.y-dD_duvw.z)/dD_duvw.x, 1.0f, 1.0f ) );
    duvw1 = normalize ( float3_t ( 1.0f, 1.0f, (-dD_duvw.x-dD_duvw.y)/dD_duvw.z ) );
  }

  float3 M[3]  = { float3_t (du.x, du.y, du.z), 
                   float3_t (dv.x, dv.y, dv.z), 
                   float3_t (dw.x, dw.y, dw.z) };
#endif



  float3 dxyz0 = M * duvw0;
  float3 dxyz1 = M * duvw1;

  float3 isonormal  = normalize ( cross ( dxyz0, dxyz1 ) );

  return float4_t(isonormal.x, isonormal.y, isonormal.z, 0.0f);
}

#endif // LIBGPUCAST_COMPUTE_ISO_NORMAL_H