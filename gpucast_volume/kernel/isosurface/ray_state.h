#ifndef LIBGPUCAST_RAYSTATE_H
#define LIBGPUCAST_RAYSTATE_H

#include <math/screen_to_object_coordinates.h>
#include <math/cross.h>
#include <math/normalize.h>
#include <math/transferfunction.h>

#include <shade/faceforward.h>

#include <isosurface/sample.h>

///////////////////////////////////////////////////////////////////////////////
struct ray_state
{
  __device__ inline
  ray_state () 
    : out_opacity ( 1.0f ),
      out_depth   ( 1.0f ),
      initialized ( false ),
      abort       ( false )
  {
    out_color = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  }

  __device__ void 
  initialize ( int2 const& coords, int width, int height, float depth, float4 const* mvp_inv, float4 const* mv_inv )
  {
    out_depth = depth;

    // determine fragment's position in object space
    float4 fragposition;
    screen_to_object_coordinates ( coords, int2_t(width, height), out_depth, mvp_inv, &fragposition );

    // determine ray origin in object space 
    origin      = mult_mat4_float4 ( mv_inv, float4_t(0.0f, 0.0f, 0.0f, 1.0f) );

    // determine and normalize ray direction
    float4 tmp  = fragposition - origin;
    direction   = normalize ( float3_t(tmp.x, tmp.y, tmp.z) );

    // compute plane representation of ray
    compute_plane_intersections ();
  }
   
  __device__ void 
  compute_plane_intersections ()
  {
    if ( fabs(direction.x) > fabs(direction.y)  && 
         fabs(direction.x) > fabs(direction.z) ) 
    {
      n1 = float3_t(direction.y, -direction.x, 0.0f);
    } else {
      n1 = float3_t(0.0f, direction.z, -direction.y);
    }

    n2 = cross(n1, direction);

    n1 = normalize(n1);
    n2 = normalize(n2);

    d1 = dot(-n1, float3_t(origin.x, origin.y, origin.z));
    d2 = dot(-n2, float3_t(origin.x, origin.y, origin.z));
  }

  __device__ inline void
  blend ( float4 const& color )
  {
    out_color             = out_color + color.w * out_opacity * float4_t(color.x, color.y, color.z, 1.0f);
    out_opacity           = out_opacity * (1.0f - color.w);
  }

  float4    out_color;

  bool      initialized;
  float4    origin;
  float3    direction;

  float3    n1, n2;
  float     d1, d2;

  float     out_opacity;   
  float     out_depth;

  bool      abort;
};

#endif // LIBGPUCAST_TARGET_FUNCTION_H