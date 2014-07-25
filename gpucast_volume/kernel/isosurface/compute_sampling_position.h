#ifndef LIBGPUCAST_COMPUTE_NEXT_SAMPLING_POSITION_H
#define LIBGPUCAST_COMPUTE_NEXT_SAMPLING_POSITION_H

#include "isosurface/target_function.h"
#include "isosurface/sample.h"
#include "isosurface/compute_sampling_position.h"

#include "math/inverse.h"
#include "math/matrix_layout.h"
#include "math/mult.h"
#include "math/transpose.h"

/////////////////////////////////////////////////////////////////////////////
__device__ inline float 
min_distance_to_boundary ( float3 const& uvw, float3 const& uvwmin, float3 const& uvwmax )
{
  float minu = min ( fabs ( uvwmin.x - uvw.x ), fabs ( uvwmax.x - uvw.x ) );
  float minv = min ( fabs ( uvwmin.y - uvw.y ), fabs ( uvwmax.y - uvw.y ) );
  float minw = min ( fabs ( uvwmin.z - uvw.z ), fabs ( uvwmax.z - uvw.z ) );
  return min(minu, min(minv, minw));
}

/////////////////////////////////////////////////////////////////////////////
__device__ inline float 
intersect_domain ( float3 const& uvw, float3 const& d, float3 const& uvwmin, float3 const& uvwmax, unsigned facemask )
{
  float t_umin = ( uvwmin.x - uvw.x ) / d.x;
  float t_umax = ( uvwmax.x - uvw.x ) / d.x;
  float t_vmin = ( uvwmin.y - uvw.y ) / d.y;
  float t_vmax = ( uvwmax.y - uvw.y ) / d.y;
  float t_wmin = ( uvwmin.z - uvw.z ) / d.z;
  float t_wmax = ( uvwmax.z - uvw.z ) / d.z;

#if 1
  bool first_intersection = true;
  float tmin = -1.0f;
  if ( (t_umin < tmin || first_intersection ) && t_umin > 0.0f && ((facemask >> 0) & 0x0001) ) { tmin = t_umin; first_intersection = false; }
  if ( (t_umax < tmin || first_intersection ) && t_umax > 0.0f && ((facemask >> 1) & 0x0001) ) { tmin = t_umax; first_intersection = false; }
  if ( (t_vmin < tmin || first_intersection ) && t_vmin > 0.0f && ((facemask >> 2) & 0x0001) ) { tmin = t_vmin; first_intersection = false; }
  if ( (t_vmax < tmin || first_intersection ) && t_vmax > 0.0f && ((facemask >> 3) & 0x0001) ) { tmin = t_vmax; first_intersection = false; }
  if ( (t_wmin < tmin || first_intersection ) && t_wmin > 0.0f && ((facemask >> 4) & 0x0001) ) { tmin = t_wmin; first_intersection = false; }
  if ( (t_wmax < tmin || first_intersection ) && t_wmax > 0.0f && ((facemask >> 5) & 0x0001) ) { tmin = t_wmax; first_intersection = false; }
#else
  float tmin = 10000000.0f;
  if ( (t_umin < tmin ) && t_umin > 0.0f ) tmin = t_umin;
  if ( (t_umax < tmin ) && t_umax > 0.0f ) tmin = t_umax;
  if ( (t_vmin < tmin ) && t_vmin > 0.0f ) tmin = t_vmin;
  if ( (t_vmax < tmin ) && t_vmax > 0.0f ) tmin = t_vmax;
  if ( (t_wmin < tmin ) && t_wmin > 0.0f ) tmin = t_wmin;
  if ( (t_wmax < tmin ) && t_wmax > 0.0f ) tmin = t_wmax;
#endif

  return tmin;
}  


/////////////////////////////////////////////////////////////////////////////
__device__ inline float 
intersect_face ( float3 const& uvw, float3 const& duvw, float pmin, float pmax, unsigned face )
{
  if ( face == 0 ) return ( pmin - uvw.x ) / duvw.x;
  if ( face == 1 ) return ( pmax - uvw.x ) / duvw.x;
  if ( face == 2 ) return ( pmin - uvw.y ) / duvw.y;
  if ( face == 3 ) return ( pmax - uvw.y ) / duvw.y;
  if ( face == 4 ) return ( pmin - uvw.z ) / duvw.z;
  if ( face == 5 ) return ( pmax - uvw.z ) / duvw.z;
  
  return 0.01f;
}  

/////////////////////////////////////////////////////////////////////////////
__device__ inline float 
intersect_domain ( float3 const& uvw, float3 const& duvw, float pmin, float pmax )
{
  float umin = ( pmin - uvw.x ) / duvw.x;
  float umax = ( pmax - uvw.x ) / duvw.x;
  float vmin = ( pmin - uvw.y ) / duvw.y;
  float vmax = ( pmax - uvw.y ) / duvw.y;
  float wmin = ( pmin - uvw.z ) / duvw.z;
  float wmax = ( pmax - uvw.z ) / duvw.z;
  
  float tmin = 1000000.0f;

  if ( umin > 0.0f ) tmin = min ( tmin, umin );
  if ( umax > 0.0f ) tmin = min ( tmin, umax );
  if ( vmin > 0.0f ) tmin = min ( tmin, vmin );
  if ( vmax > 0.0f ) tmin = min ( tmin, vmax );
  if ( wmin > 0.0f ) tmin = min ( tmin, wmin );
  if ( wmax > 0.0f ) tmin = min ( tmin, wmax );
  
  return tmin;
}  


/////////////////////////////////////////////////////////////////////////////
template <typename ray_t>
__device__ inline
float3 transform_ray_to_domain_space ( ray_t  const& ray_direction,
                                       float4 const& du,
                                       float4 const& dv,
                                       float4 const& dw )
{
  // the partial derivatives dV/du, dV/dv and dV/dw constitute the coordinate system V
  float3 V[3] = { float3_t(du.x, du.y, du.z), 
                  float3_t(dv.x, dv.y, dv.z),  
                  float3_t(dw.x, dw.y, dw.z) };

  float3 Vinv[3]; 
  inverse3 ( V, Vinv );

  // transform ray to partial derivative system -> duvw_dt
  float3 duvw_dt           = mult_mat3_float3 ( Vinv, float3_t ( ray_direction.x, ray_direction.y, ray_direction.z) );
  return duvw_dt;
}


/////////////////////////////////////////////////////////////////////////////
__device__ inline
float4 compute_sampling_position_for_domain_intersection ( float4 const& last_sample_position,
                                                           float4 const& ray_direction,
                                                           float         max_sample_distance,
                                                           float         sample_distance_object_space,
                                                           float         pixel_error_object_space,
                                                           float3 const& last_sample_uvw,
                                                           float4 const& du, 
                                                           float4 const& dv, 
                                                           float4 const& dw,
                                                           unsigned      surface_type )
{
  // transform ray to domain
  float3 duvw_dt = transform_ray_to_domain_space ( ray_direction, du, dv, dw );

  // slow down step to avoid artifacts by hitting face directly
  float step_abberation    = 1.0f + 0.5f * pixel_error_object_space;
  
  // compute newton step towards isoparametric face
  float t_intersect        = step_abberation * intersect_face   ( last_sample_uvw, duvw_dt, 0.0f, 1.0f, surface_type );

  // clamp step to pixel length in object space and percentage of total sample distance
  float step_length_domain = max ( pixel_error_object_space, min ( max_sample_distance * sample_distance_object_space, t_intersect ) );

  return last_sample_position + step_length_domain * ray_direction;
}


/////////////////////////////////////////////////////////////////////////////
__device__ inline
float4 compute_sampling_position_adaptively ( float4 const& last_sample_position,
                                              float4 const& ray_direction,
                                              float         obb_diameter,
                                              float         min_sample_distance,
                                              float         max_sample_distance,
                                              float         adaptive_sample_scale,
                                              float         iso_value,
                                              float3 const& last_sample_uvw,
                                              float2 const& last_sample_data,
                                              float4 const& du, 
                                              float4 const& dv, 
                                              float4 const& dw,
                                              float2 const& ddu, 
                                              float2 const& ddv, 
                                              float2 const& ddw )
{
  // partial data derivatives: dD/du, dD/dv, dD/dw transformed into one-dimensional iso space
  float3 dD_duvw  = float3_t ( ddu.x, 
                               ddv.x,
                               ddw.x );

  float3 duvw_dt = transform_ray_to_domain_space ( ray_direction, du, dv, dw );

  // how does attribute change along the extrapolation along the ray
  float dD_dt    = dot ( duvw_dt, dD_duvw ); 

  // compute distance to isosurface in data space
  float dD0       = target_function(iso_value) - target_function(last_sample_data.x);

  float adaptive_stepwidth = 0.0;
  
  // compute step
  if ( dD_dt * dD0 < 0.0 ) 
  {
    // ray and isosurface diverge -> go max step according to bbox diameter
    adaptive_stepwidth    =  obb_diameter * max_sample_distance;
  } else {
    // ray and isosurface seem to converge -> try to solve equation assuming linearity
    adaptive_stepwidth    =  adaptive_sample_scale * dD0 / dD_dt ;
  }

  // clamp stepwidth
  adaptive_stepwidth       = clamp ( adaptive_stepwidth, obb_diameter * min_sample_distance, obb_diameter * max_sample_distance );
 
  // compute next sample position
  return last_sample_position + ray_direction * adaptive_stepwidth;
}


/////////////////////////////////////////////////////////////////////////////
__device__ inline
float4 compute_sampling_position_adaptively ( float4 const& last_sample_position,
                                              float4 const& ray_direction,
                                              float         obb_diameter,
                                              float         min_sample_distance,
                                              float         max_sample_distance,
                                              float         adaptive_sample_scale,
                                              float         iso_value,
                                              sample const& last_sample,
                                              bool          clamp_outer,
                                              unsigned      facemask )
{
  // transform ray to domain
  float3 duvw_dt = transform_ray_to_domain_space ( ray_direction, last_sample.dp_du, last_sample.dp_dv, last_sample.dp_dw );

  // partial data derivatives: dD/du, dD/dv, dD/dw transformed into one-dimensional iso space
  float3 dD_duvw  = float3_t ( last_sample.da_du.x, 
                               last_sample.da_dv.x,
                               last_sample.da_dw.x );

  // how does attribute change along the extrapolation along the ray
  float dD_dt    = dot ( duvw_dt, dD_duvw ); 
  if ( dD_dt == 0.0f )
  {
    dD_dt = 0.0001f;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // compute distance to isosurface in data space and try to "hit" isosurface
  ///////////////////////////////////////////////////////////////////////////////////////
  float dD0      = iso_value - last_sample.a.x;

  float adaptive_stepwidth = 0.0;
  
  // compute step
  if ( dD_dt * dD0 < 0.0 ) 
  {
    // ray and isosurface diverge -> go max step according to bbox diameter
    adaptive_stepwidth    =  obb_diameter * max_sample_distance;
  } else {
    // ray and isosurface seem to converge -> try to solve equation assuming linearity
    adaptive_stepwidth    =  adaptive_sample_scale * dD0 / dD_dt ;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  // if there is an intersection with domain boundary -> clamp to intersection
  ///////////////////////////////////////////////////////////////////////////////////////
  if ( clamp_outer ) 
  {
    float3 target_uvw = last_sample.uvw + adaptive_stepwidth * duvw_dt;

    // clamp to domain intersection
#if 0
    float3 dense_sample_domain_min = float3_t(max_sample_distance, max_sample_distance, max_sample_distance);
    float3 dense_sample_domain_max = float3_t(1.0f-max_sample_distance, 1.0f-max_sample_distance, 1.0f-max_sample_distance);

    if ( !in_domain ( target_uvw, dense_sample_domain_min, dense_sample_domain_max) ) //||
         //!in_domain ( last_sample.uvw, dense_sample_domain_min, dense_sample_domain_max))
    {
      float t_to_boundary_intersection = intersect_domain ( last_sample.uvw, 
                                                            adaptive_stepwidth * duvw_dt, 
                                                            float3_t(0.0f, 0.0f, 0.0f), 
                                                            float3_t(1.0f, 1.0f, 1.0f),
                                                            facemask );

      if ( t_to_boundary_intersection > 0.0f )
      {
        // clamp step to intersection with linear extrapolation
        adaptive_stepwidth = adaptive_sample_scale * t_to_boundary_intersection;
      }
    }
#endif

    // clamp to minimal distance to boundary
#if 1
    
    float current_min_dist = min_distance_to_boundary ( last_sample.uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));
    float next_min_dist    = min_distance_to_boundary ( target_uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));

    float min_dist         = min ( fabs( current_min_dist), fabs(next_min_dist) );

    adaptive_stepwidth     = min_dist;
   
#endif

    // clamp to distance if angle is too low ->
#if 0
    float3 step   = normalize ( target_uvw - last_sample.uvw );

    float alpha_u = step.x;
    float alpha_v = step.y;
    float alpha_w = step.z;
    
    float current_min_dist = min_distance_to_boundary ( last_sample.uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));
    float next_min_dist    = min_distance_to_boundary ( target_uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));

    float min_dist         = min ( fabs( current_min_dist), fabs(next_min_dist));

    if ( alpha_u > 0.8f || alpha_v > 0.8f || alpha_w > 0.8f )
    {
      adaptive_stepwidth     = adaptive_sample_scale * min (adaptive_stepwidth, min_dist);
    }
#endif
    ///////////////////////////////////////////////////////////////////////////////////////
    // if sampling is close to boundary -> be careful 
    ///////////////////////////////////////////////////////////////////////////////////////
    //float distance_to_boundary = min_distance_to_boundary( last_sample.uvw, float3_t(0.0f, 0.0f, 0.0f), float3_t(1.0f, 1.0f, 1.0f));
    //adaptive_stepwidth         = adaptive_sample_scale * distance_to_boundary;
  }

  // clamp stepwidth
  adaptive_stepwidth       = clamp ( adaptive_stepwidth, obb_diameter * min_sample_distance, obb_diameter * max_sample_distance );
 
  // compute next sample position
  return last_sample_position + ray_direction * adaptive_stepwidth;
}


/////////////////////////////////////////////////////////////////////////////
__device__ inline
float4 compute_sampling_position ( float4   last_sample_position,
                                   float4   ray_direction,
                                   float    min_sample_distance,
                                   float    obb_diameter )
{
  float4 fixed_stepwidth = ray_direction * obb_diameter * min_sample_distance;
  return last_sample_position + fixed_stepwidth;
}

#endif