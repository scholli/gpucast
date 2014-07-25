#ifndef LIB_GPUCAST_NEWTON_VOLUME_H
#define LIB_GPUCAST_NEWTON_VOLUME_H

#include "./local_memory_config.h"
#include "math/horner_volume.h"
#include "math/in_domain.h"

/**********************************************************************
* newton iteration to iterate to determine the parameter values for a sample point
***********************************************************************/
__device__
inline bool
newton_volume (float4 const* points,
               int           baseid,
               uint3 const&  order,
               float3 const& in_uvw,
               float3&       uvw,
               float4 const& target_point,
               float4&       point,
               float4&       du,
               float4&       dv,
               float4&       dw,
               float3 const& n1, 
               float3 const& n2,
               float3 const& n3,
               float         d1, 
               float         d2,
               float         epsilon,
               int           maxsteps )
{
  // init
  point = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  du    = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  dv    = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  dw    = float4_t(0.0f, 0.0f, 0.0f, 0.0f);

  // start iteration at in_uvw
  uvw = in_uvw;

  // set target point in volume
  float d3    = dot(-1.0f * n3, float3_t(target_point.x, target_point.y, target_point.z));

	for (int i = 1; i != maxsteps; ++i)
  {
    // evaluate volume at starting point
    horner_volume_derivatives<float4, 3> ( points, baseid, order, uvw, point, du, dv, dw);
    
		// compute distance to planes -> stepwidth
		float3 Fuvw = float3_t(dot(float3_t(point.x, point.y, point.z), n1) + d1, 
                           dot(float3_t(point.x, point.y, point.z), n2) + d2, 
                           dot(float3_t(point.x, point.y, point.z), n3) + d3);
	
    // abort criteria I: approximate intersection reached
    if (length(Fuvw) < epsilon) {
		  break;
    }
    
		// map base vectors to planes
		float3 Fu   = float3_t( dot(n1, float3_t(du.x, du.y, du.z)), dot(n2, float3_t(du.x, du.y, du.z)), dot(n3, float3_t(du.x, du.y, du.z)));
		float3 Fv   = float3_t( dot(n1, float3_t(dv.x, dv.y, dv.z)), dot(n2, float3_t(dv.x, dv.y, dv.z)), dot(n3, float3_t(dv.x, dv.y, dv.z)));
		float3 Fw   = float3_t( dot(n1, float3_t(dw.x, dw.y, dw.z)), dot(n2, float3_t(dw.x, dw.y, dw.z)), dot(n3, float3_t(dw.x, dw.y, dw.z)));
	
#ifdef ROW_MAJOR
    float3 J[3] = { float3_t(Fu.x, Fv.x, Fw.x),
                    float3_t(Fu.y, Fv.y, Fw.y),
                    float3_t(Fu.z, Fv.z, Fw.z) };
#endif

#ifdef COL_MAJOR
    float3 J[3] = {Fu, Fv, Fw}; 
#endif

    // stop iteration if there's no solution 
    float Jdet = determinant3(J);
    if ( Jdet == 0.0 ) 
    {
      break;
    }

    float3 Jinv[3];
    inverse3(J, Jinv);

    // do step in parameter space
		uvw       = uvw - mult_mat3_float3 (Jinv, Fuvw);

    // clamp result to regular domain
    uvw       = clamp(uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f) );
  }
  
  // clamp result to regular domain
  //*uvw       = clamp(*uvw, (float3)(0,0,0), (float3)(1,1,1) );
  
  float3 error = float3_t( dot(float3_t(point.x, point.y, point.z), n1) + d1, 
                           dot(float3_t(point.x, point.y, point.z), n2) + d2, 
                           dot(float3_t(point.x, point.y, point.z), n3) + d3);

  // success criteria -> point near sample & parameter in domain
	return ( length(error) <= epsilon );
}


/**********************************************************************
* newton iteration to iterate to determine the parameter values for a sample point
***********************************************************************/
__device__
inline bool
newton_volume_unbound ( 
               float4 const*          points,
               int                    baseid,
               uint3 const&           order,
               float3 const&          in_uvw,
               float3&                uvw,
               float4 const&          target_point,
               float4&                point,
               float4&                du,
               float4&                dv,
               float4&                dw,
               float3 const&          n1, 
               float3 const&          n2,
               float3 const&          n3,
               float                  d1, 
               float                  d2,
               float                  epsilon,
               int                    maxsteps
               )
{
  // init
  point = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  du    = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  dv    = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  dw    = float4_t(0.0f, 0.0f, 0.0f, 0.0f);

  // start iteration at in_uvw
  uvw = in_uvw;

  // set target point in volume
  float d3    = dot(-1.0f * n3, float3_t(target_point.x, target_point.y, target_point.z));

	for (int i = 1; i != maxsteps; ++i)
  {
    // evaluate volume at starting point
    horner_volume_derivatives<float4, 3> ( points, baseid, order, uvw, point, du, dv, dw);
    
		// compute distance to planes -> stepwidth
		float3 Fuvw = float3_t(dot(float3_t(point.x, point.y, point.z), n1) + d1, 
                           dot(float3_t(point.x, point.y, point.z), n2) + d2, 
                           dot(float3_t(point.x, point.y, point.z), n3) + d3);
	
    // abort criteria I: approximate intersection reached
    if (length(Fuvw) < epsilon) {
		  break;
    }
    
		// map base vectors to planes
		float3 Fu   = float3_t( dot(n1, float3_t(du.x, du.y, du.z)), dot(n2, float3_t(du.x, du.y, du.z)), dot(n3, float3_t(du.x, du.y, du.z)));
		float3 Fv   = float3_t( dot(n1, float3_t(dv.x, dv.y, dv.z)), dot(n2, float3_t(dv.x, dv.y, dv.z)), dot(n3, float3_t(dv.x, dv.y, dv.z)));
		float3 Fw   = float3_t( dot(n1, float3_t(dw.x, dw.y, dw.z)), dot(n2, float3_t(dw.x, dw.y, dw.z)), dot(n3, float3_t(dw.x, dw.y, dw.z)));
	
#ifdef ROW_MAJOR
    float3 J[3] = { float3_t(Fu.x, Fv.x, Fw.x),
                    float3_t(Fu.y, Fv.y, Fw.y),
                    float3_t(Fu.z, Fv.z, Fw.z) };
#endif

#ifdef COL_MAJOR
    float3 J[3] = {Fu, Fv, Fw}; 
#endif

    // stop iteration if there's no solution 
    float Jdet = determinant3(J);
    if ( Jdet == 0.0 ) 
    {
      break;
    }

    float3 Jinv[3];
    inverse3(J, Jinv);

    // do step in parameter space
		uvw       = uvw - mult_mat3_float3 (Jinv, Fuvw);
  }
  
  float3 error = float3_t( dot(float3_t(point.x, point.y, point.z), n1) + d1, 
                           dot(float3_t(point.x, point.y, point.z), n2) + d2, 
                           dot(float3_t(point.x, point.y, point.z), n3) + d3);

  // success criteria -> point near sample & parameter in domain
	return ( length(error) <= epsilon );
}




#endif // LIB_GPUCAST_NEWTON_VOLUME_H