#ifndef LIBGPUCAST_BINARY_SEARCH_FOR_ISOSURFACE_H
#define LIBGPUCAST_BINARY_SEARCH_FOR_ISOSURFACE_H

#include "./local_memory_config.h"

#include "isosurface/target_function.h"
#include "isosurface/compute_iso_normal.h"
#include "isosurface/ray_state.h"
#include "isosurface/sample.h"

#include "math/newton_volume.h"
#include "math/horner_volume.h"

/////////////////////////////////////////////////////////////////////////////
__device__ inline
void binary_search_for_isosurface ( float4 const*     volumepointsbuffer,
                                    float2 const*     attributepointsbuffer,
                                    int               point_base_id,
                                    int               attribute_base_id,
                                    uint3 const&      order, 
                                    float4 const&     ray_entry,
                                    float4 const&     ray_direction,
                                    float3 const&     first_uvw, 
                                    float3 const&     last_uvw, 
                                    float4 const&     first_sample_position, 
                                    float4 const&     last_sample_position, 
                                    float2 const&     first_sample_data, 
                                    float2 const&     last_sample_data, 
                                    float             iso_threshold,
                                    float             newton_epsilon,
                                    unsigned          newton_max_iterations,
                                    unsigned          max_steps_binary_search,
                                    float             d1,
                                    float             d2,
                                    float3 const&     n1,
                                    float3 const&     n2,
                                    float3*           iso_uvw, 
                                    float4*           iso_position, 
                                    float2*           iso_data,
                                    float4*           iso_normal,
                                    unsigned&         total_samples )
{
  
#if 0
  
  // approximate binary search by using a simle linear interpolation
  float interp  = (target_function(iso_threshold) - target_function(first_sample_data)) / (target_function(last_sample_data) - target_function(first_sample_data));

  *iso_data      = mix(last_sample_data,     first_sample_data,     interp);
  *iso_uvw       = mix(last_uvw,             first_uvw,             interp);
  *iso_position  = mix(last_sample_position, first_sample_position, interp);
#else

  // binary search the corresponding uvw-interval -> ignore geometric position
  int  iterations   = 0;

  float3 uvwstart     = mix(first_uvw, last_uvw, 0.5f);
  float4 point        = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  float4 p, pdu, pdv, pdw;
  float2 attrib, adu, adv, adw;

  float4 pmin         = first_sample_position;
  float4 pmax         = last_sample_position;

  bool increasing_target_function = target_function ( first_sample_data.x ) < target_function ( last_sample_data.x );
  bool abort_bisection            = false;

  while ( iterations < max_steps_binary_search && !abort_bisection )
  {
    ++total_samples;

    point = mix(pmin, pmax, 0.5f); 
    float3 uvw;

    newton_volume ( volumepointsbuffer, 
                    point_base_id,
                    order,
                    uvwstart,
                    uvw,
                    point,
                    p,
                    pdu,
                    pdv,
                    pdw,
                    n1,
                    n2,
                    float3_t(ray_direction.x, ray_direction.y, ray_direction.z),
                    d1,
                    d2,
                    newton_epsilon,
                    newton_max_iterations );

    horner_volume_derivatives<float2, 1> ( attributepointsbuffer, attribute_base_id, order, uvw, attrib, adu, adv, adw );

    if ( target_function ( attrib.x ) < target_function ( iso_threshold ) == increasing_target_function ) 
    {
      pmin = point;
    } else {
      pmax = point;
    }

    uvwstart = uvw;

    ++iterations;

    float4 max_error = pmax - pmin;
    max_error.w      = 0.0;
    abort_bisection  = length(max_error) < newton_epsilon;
  } 

  *iso_uvw      = uvwstart;
  *iso_data     = attrib;
  *iso_position = point;
  *iso_normal   = compute_iso_normal ( pdu, pdv, pdw, adu, adv, adw );
                   
#endif
}

/////////////////////////////////////////////////////////////////////////////
__device__ inline
void binary_search_for_isosurface ( float4 const*     volumepointsbuffer,
                                    float2 const*     attributepointsbuffer,
                                    int               point_base_id,
                                    int               attribute_base_id,
                                    sample const&     fst_sample,
                                    sample const&     snd_sample,
                                    sample&           iso_sample,
                                    ray_state const&  ray,
                                    float             iso_threshold,
                                    float             newton_epsilon,
                                    unsigned          newton_max_iterations,
                                    unsigned          max_steps_binary_search,
                                    unsigned&         total_samples
                                  )
{
  
  // binary search the corresponding uvw-interval -> ignore geometric position
  int  iterations   = 0;

  float3 uvwstart     = mix(fst_sample.uvw, snd_sample.uvw, 0.5f);
  float4 point        = float4_t(0.0f, 0.0f, 0.0f, 0.0f);
  
  float4 pmin         = fst_sample.p;
  float4 pmax         = snd_sample.p;

  bool increasing_target_function = target_function ( fst_sample.a.x ) < target_function ( snd_sample.a.x );
  bool abort_bisection            = false;

  while ( iterations < max_steps_binary_search && 
          !abort_bisection )
  {
    ++total_samples;

    point = mix(pmin, pmax, 0.5f); 
    iso_sample.inversion_success = newton_volume_unbound ( volumepointsbuffer, 
                                                           point_base_id,
                                                           fst_sample.volume.volume_order,
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
                                           attribute_base_id, 
                                           fst_sample.volume.volume_order, 
                                           iso_sample.uvw, 
                                           iso_sample.a, 
                                           iso_sample.da_du, 
                                           iso_sample.da_dv, 
                                           iso_sample.da_dw );

    if ( target_function ( iso_sample.a.x ) < target_function ( iso_threshold ) == increasing_target_function ) 
    {
      pmin = point;
    } else {
      pmax = point;
    }

    uvwstart = iso_sample.uvw;

    ++iterations;

    float4 max_error = pmax - pmin;
    max_error.w      = 0.0;
    abort_bisection  = length(max_error) < newton_epsilon;
  }     
}



#endif // LIBGPUCAST_BINARY_SEARCH_FOR_ISOSURFACE_H