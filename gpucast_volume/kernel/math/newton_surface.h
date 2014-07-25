#ifndef LIB_GPUCAST_NEWTON_SURFACE_H
#define LIB_GPUCAST_NEWTON_SURFACE_H

#include <math/determinant.h>
#include <math/horner_surface.h>
#include <math/inverse.h>
#include <math/mult.h>
#include <math/length.h>
#include <math/dot.h>

/**********************************************************************
* newton iteration to find surface intersection
***********************************************************************/
__device__
inline bool
newton_surface( float4 const*  points,
                int            baseid,
                float2&        uv,
                float          epsilon,
                int            maxsteps,
                uint2          order,
                float3 const&  n1, 
                float3 const&  n2,
                float          d1, 
                float          d2,
                float4&        p, 
                float4&        du, 
                float4&        dv,
                unsigned&      iterations )
{
  float2 Fuv = float2_t(0.0f, 0.0f);

  for (int i = 0; i < maxsteps; ++i) 
  {
    ++iterations;

    horner_surface_derivatives ( points, baseid, order, uv, p, du, dv );

    Fuv         = float2_t(dot(n1, float3_t(p.x, p.y, p.z)) + d1, dot(n2, float3_t(p.x, p.y, p.z)) + d2);
    
    float2 Fu   = float2_t(dot(n1, float3_t(du.x, du.y, du.z)), dot(n2, float3_t(du.x, du.y, du.z)));  
    float2 Fv   = float2_t(dot(n1, float3_t(dv.x, dv.y, dv.z)), dot(n2, float3_t(dv.x, dv.y, dv.z)));  

#ifdef ROW_MAJOR
    float2 J[2] = { float2_t(Fu.x, Fv.x),
                    float2_t(Fu.y, Fv.y) };
#endif

#ifdef COL_MAJOR
    float2 J[2] = {Fu, Fv}; 
#endif

    float2 Jinv[2];
    inverse2 (J, Jinv);

    uv = uv - mult_mat2_float2 ( Jinv, Fuv ); 

    if (length(Fuv) < epsilon) {
      break;
    }
  } 

  // return if convergence was reached
  return !(length(Fuv) > epsilon || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0);
}

#endif // LIB_GPUCAST_NEWTON_SURFACE_H