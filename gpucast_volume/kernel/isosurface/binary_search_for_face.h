#ifndef LIBGPUCAST_BINARY_SEARCH_FOR_FACE_H
#define LIBGPUCAST_BINARY_SEARCH_FOR_FACE_H

#include "./local_memory_config.h"

#include "isosurface/target_function.h"
#include "isosurface/compute_iso_normal.h"
#include "isosurface/ray_state.h"
#include "isosurface/sample.h"

#include "math/newton_volume.h"
#include "math/horner_volume.h"

/////////////////////////////////////////////////////////////////////////////
__device__ inline
void binary_search_for_face ( float4 const*     volumepointsbuffer,
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
  
  // binary search the corresponding uvw-interval -> ignore geometric position
  int  iterations     = 0;

  float3 uvwmid       = mix(fst_sample.uvw, snd_sample.uvw, 0.5f);
  float4 point        = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  
  float4 pmin         = fst_sample.p;
  float4 pmax         = snd_sample.p;

  bool abort_bisection            = false;

  while ( iterations < max_steps_binary_search && 
          !abort_bisection )
  {
    point = mix(pmin, pmax, 0.5f); 
    
    iso_sample.inversion_success = newton_volume_unbound ( volumepointsbuffer, 
                                                           volume_points_id,
                                                           fst_sample.volume.volume_order,
                                                           uvwmid,
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
                                           fst_sample.volume.volume_order, 
                                           iso_sample.uvw, 
                                           iso_sample.a, 
                                           iso_sample.da_du, 
                                           iso_sample.da_dv, 
                                           iso_sample.da_dw );

    if ( iso_sample.uvw_in_domain() != fst_sample.uvw_in_domain() ) 
    {
      pmax = point;
    } else {
      pmin = point;
    }

    uvwmid     = iso_sample.uvw;

    ++iterations;

    float4 max_error = pmax - pmin;
    max_error.w      = 0.0;
    abort_bisection  = length(max_error) < newton_epsilon;
  }
}



#endif // LIBGPUCAST_BINARY_SEARCH_FOR_ISOSURFACE_H