#ifndef LIBGPUCAST_NEWTON_SEARCH_FOR_ISOSURFACE_H
#define LIBGPUCAST_NEWTON_SEARCH_FOR_ISOSURFACE_H

#include "./local_memory_config.h"

#include "isosurface/target_function.h"
#include "isosurface/compute_iso_normal.h"
#include "math/newton_volume.h"
#include "math/horner_volume.h"
#include "math/transpose.h"

/////////////////////////////////////////////////////////////////////////////
__device__ inline 
void newton_search_for_isosurface ( float4 const*     volumepointsbuffer,
                                    float2 const*     attributepointsbuffer,
                                    int               volume_base_id,
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
                                    float4*           iso_normal )
{
  // binary search the corresponding uvw-interval -> ignore geometric position
  int  iterations   = 0;

  // start in middle of interval
  float4 sample_point = mix ( first_sample_position, last_sample_position, 0.5f );
  float3 uvw          = mix ( first_uvw, last_uvw, 0.5f );
  float4 point, pdu, pdv, pdw;
  float2 attrib, adu, adv, adw;

  // compute maximum sampling step distance
  float  sample_range   = length (float3_t(last_sample_position.x, last_sample_position.y, last_sample_position.z) - float3_t(first_sample_position.x, first_sample_position.y, first_sample_position.z));

  // start iteration -> TODO: other abort criteria!
  while ( iterations < max_steps_binary_search )
  {
    // iterate to sample to determine according uvw value
    float3 uvw_next;

    newton_volume ( volumepointsbuffer, 
                    volume_base_id,
                    order,
                    uvw,
                    uvw_next,
                    sample_point,
                    point,
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

    uvw = uvw_next;

    //horner_volume_derivatives ( pointbuffer,     point_base_id,     order, uvw, &point,  &pdu, &pdv, &pdw );
    horner_volume_derivatives<float2, 1> ( attributepointsbuffer, attribute_base_id, order, uvw, attrib, adu, adv, adw );

    #ifdef ROW_MAJOR
    float3 V[3] = { (float3)(pdu.x, pdv.x, pdw.x),
                    (float3)(pdu.y, pdv.y, pdw.y),
                    (float3)(pdu.z, pdv.z, pdw.z) };
    #endif

    #ifdef COL_MAJOR
    float3 V[3] = { float3_t(pdu.x, pdu.x, pdu.x),
                    float3_t(pdv.x, pdv.y, pdv.z), 
                    float3_t(pdw.x, pdw.y, pdw.z) };
    #endif

    float3 Vinv[3]; 
    inverse3   ( V, Vinv );

    float3 VinvT[3]; 
    transpose3 ( Vinv, VinvT );

    // partial data derivatives: dattrib/du, dattrib/dv, dattrib/dw transformed into one-dimensional iso space
    float3 dD_duvw  = float3_t ( target_function(adu.x), 
                                 target_function(adv.x),
                                 target_function(adw.x) );

    // transform data derivative into object space -> how much does D change if you go along the ray (dx,dy,dz)
    float3 dD_dxyz  = mult_mat3_float3 ( VinvT, dD_duvw );

    // dot product with ray indicates -> how does D change if you step 1 along the ray
    float dD_dt     = dot ( dD_dxyz, float3_t(ray_direction.x, ray_direction.y, ray_direction.z));

    // compute distance to isosurface in data space
    float dD0       = target_function(iso_threshold) - target_function(attrib.x);

    // ray and isosurface seem to converge -> try to solve equation assuming linearity
    float adaptive_stepwidth = dD0 / dD_dt ;

    // compute next sample position
    sample_point = sample_point + ray_direction * adaptive_stepwidth;

    ++iterations;
  } 

  *iso_uvw      = uvw;
  *iso_data     = attrib;
  *iso_position = point;
  *iso_normal   = compute_iso_normal ( pdu, pdv, pdw, adu, adv, adw );
}

#endif // LIBGPUCAST_BINARY_SEARCH_FOR_ISOSURFACE_H