#ifndef LIBGPUCAST_SECANT_SEARCH_FOR_ISOSURFACE_H
#define LIBGPUCAST_SECANT_SEARCH_FOR_ISOSURFACE_H

#include "./local_memory_config.h"

#include "isosurface/target_function.h"
#include "isosurface/compute_iso_normal.h"
#include "math/newton_volume.h"
#include "math/horner_volume.h"

/////////////////////////////////////////////////////////////////////////////
__device__ inline
void secant_search_for_face ( float4 const*     volumepointsbuffer,
                              float2 const*     attributepointsbuffer,
                              unsigned          volume_points_id,
                              unsigned          attribute_points_id,
                              sample const&     fst_sample,
                              sample const&     snd_sample,
                              sample&           iso_sample,
                              ray_state const&  ray,
                              float             iso_threshold,
                              float             newton_epsilon,
                              unsigned          newton_max_iterations,
                              unsigned          max_steps_binary_search )
{
  // secant search the corresponding uvw-interval -> ignore geometric position
  int  iterations   = 0;
  float fst_a = fst_sample.a.x;
  float snd_a = snd_sample.a.x;

  float interp                    = (iso_threshold - fst_a) / (snd_a - fst_a);
  bool increasing_target_function = fst_a < snd_a;

  float3 uvwstart     = mix(fst_sample.uvw, snd_sample.uvw, interp);
  float4 point        = float4_t(0.0f, 0.0f, 0.0f, 0.0f);

  float4 pmin         = fst_sample.p;
  float4 pmax         = snd_sample.p;

  while ( iterations < max_steps_binary_search )
  {
    interp  = (iso_threshold - fst_a) / (snd_a - fst_a);
    point  = mix(pmin, pmax, interp); 
    
    newton_volume ( volumepointsbuffer, 
                    volume_points_id,
                    iso_sample.volume.volume_order,
                    uvwstart,
                    iso_sample.uvw,
                    point,
                    iso_sample.p,
                    iso_sample.dp_du,
                    iso_sample.dp_dv,
                    iso_sample.dp_dw,
                    ray.n1,
                    ray.n2,
                    ray.direction,
                    ray.d1,
                    ray.d2,
                    newton_epsilon,
                    newton_max_iterations );

    horner_volume_derivatives<float2, 1> ( attributepointsbuffer, 
                                           attribute_points_id, 
                                           iso_sample.volume.volume_order, 
                                           iso_sample.uvw, 
                                           iso_sample.a,
                                           iso_sample.da_du, 
                                           iso_sample.da_dv, 
                                           iso_sample.da_dw );

    if ( iso_sample.a.x < iso_threshold == increasing_target_function ) 
    {
      pmin   = point;
      fst_a  = iso_sample.a.x;
    } else {
      pmax   = point;
      snd_a  = iso_sample.a.x;
    }

    if ( length ( iso_sample.uvw - uvwstart) < newton_epsilon )
    {
      break;
    }

    uvwstart = iso_sample.uvw;

    ++iterations;
  } 
}

#endif // LIBGPUCAST_SECANT_SEARCH_FOR_ISOSURFACE_H