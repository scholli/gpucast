#ifndef LIBGPUCAST_RAYCAST_SURFACES_IN_FRAGMENTLIST_H
#define LIBGPUCAST_RAYCAST_SURFACES_IN_FRAGMENTLIST_H

#include <math/compute_depth.h>
#include <math/newton_surface.h>
#include <math/clamp.h>
#include <math/in_domain.h>
#include <math/mix.h>
#include <math/ray_to_plane_intersection.h>
#include <math/screen_to_object_coordinates.h>

#include <fragmentlist/bubble_sort.h>
#include <isosurface/compute_sampling_position.h>
#include <isosurface/fragment.h>
#include <isosurface/sample.h>
#include <fragmentlist/classify_intersection.h>

#define MINIMAL_SAMPLING_DISTANCE                         0.001f

#define INTERSECT_THIN_CONVEX_HULLS_EPSILON_OBJECT_SPACE  0.03f
#define INTERSECT_THIN_CONVEX_HULLS_ONCE                  1

#define ADAPTIVE_SAMPLING                                 1

#define INTERSECT_ON_PARTIAL_DERIVATIVE_CHANGE            0

#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279f
#endif


///////////////////////////////////////////////////////////////////////////////
__device__ inline
void compute_face_intersections_per_fragment ( int           start_index, 
                                               int           nfragments,
                                               unsigned&     nface_intersections,
                                               unsigned&     nface_intersection_tests,
                                               int2 const&   screen_coords,
                                               int2 const&   screen_resolution,
                                               float         surface_transparency, 
                                               float         fixed_newton_epsilon,
                                               unsigned      max_newton_iterations,
                                               float         nearplane,
                                               float         farplane,
                                               uint4*        indexlist,
                                               uint4 const*  surfacedatabuffer,
                                               float4 const* surfacepointsbuffer,
                                               float4 const* modelviewmatrix,
                                               float4 const* modelviewmatrixinverse,
                                               float4 const* modelviewprojectionmatrixinverse )
{
  if ( nfragments == 0 ) { 
    return;
  }

  /********************************************************************************
  * allocate variables for ray casting result
  ********************************************************************************/ 
  int    fragindex                   = start_index;

  /********************************************************************************
  * init out color and transparency
  ********************************************************************************/ 
  float out_opacity                  = 1.0f;

/********************************************************************************
  * traverse depth-sorted fragments
  ********************************************************************************/ 
  for ( int i = 0; i != nfragments; ++i )
  {
    /********************************************************************************
    * quit shading when alpha=1.0 is reached
    ********************************************************************************/ 
    if ( out_opacity < 0.01 )
    {
      nfragments = i;
      break;
    }

    /********************************************************************************
    * retrieve information about fragment, volume and surface
    ********************************************************************************/ 
    uint4    fragindexinfo      = indexlist [ fragindex ];
    unsigned surface_id         = fragindexinfo.z;

    /********************************************************************************
    * unpack data for intersection
    ********************************************************************************/ 
    uint4   surface_info       = surfacedatabuffer[surface_id + 1];
    uint4   surface_order_info = surfacedatabuffer[surface_id + 2];

    float2  uv                 = unpack_uv ( fragindexinfo.y );
    uv                         = clamp ( uv, float2_t(0.0f, 0.0f), float2_t(1.0f, 1.0f));

    uint2   order              = uint2_t ( surface_order_info.y, surface_order_info.z );
    unsigned surface_mesh_id   = surface_info.y;

    /********************************************************************************
    * ray setup
    ********************************************************************************/ 
    float3 n1, n2;
    float  d1, d2;

    float4  fragposition;

    screen_to_object_coordinates ( screen_coords, screen_resolution, intBitsToFloat ( fragindexinfo.w ), modelviewprojectionmatrixinverse, &fragposition);

    point_to_plane_intersection ( fragposition, modelviewmatrixinverse, &n1, &n2, &d1, &d2 );

    /********************************************************************************
    * intersect surface
    ********************************************************************************/ 
    float newton_epsilon = fixed_newton_epsilon;
    float4 p, du, dv;
    unsigned iterations = 0;

    ++nface_intersection_tests;
    bool surface_intersection = newton_surface ( surfacepointsbuffer, surface_mesh_id, uv, newton_epsilon, max_newton_iterations, order, n1, n2, d1, d2, p, du, dv, iterations);

    /********************************************************************************
    * update fragment entry
    ********************************************************************************/ 
    if ( surface_intersection )
    {
      ++nface_intersections;
      // update intersection uv and depth
      float fdepth                     = compute_depth_from_object_coordinates ( modelviewmatrix, p, nearplane, farplane ); 
      indexlist[int_t(fragindex)]      = uint4_t(fragindexinfo.x, pack_uv(uv), fragindexinfo.z, floatBitsToInt(fdepth));   
    } else {
      indexlist[int_t(fragindex)]      = uint4_t(fragindexinfo.x, pack_uv(float2_t(-1.0f, -1.0f)), fragindexinfo.z, fragindexinfo.w);   
    }

    // go to next fragment
    fragindex = int_t(fragindexinfo.x);
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
struct oriented_point 
{
  float4 p;
  float3 uvw;

  float4 du;
  float4 dv;
  float4 dw;
};
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
__device__ inline
float2 interpolate_uv_guess ( float3 const& uvw_first, float3 const& uvw_last, unsigned surface_type, float epsilon )
{
  float3 d = uvw_last - uvw_first;

  float3 t_min = (float3_t(0.0, 0.0, 0.0) - uvw_first ) / d;
  float3 t_max = (float3_t(1.0, 1.0, 1.0) - uvw_first ) / d;

  if ( surface_type == 0 ) return float2_t (uvw_first.y + t_min.x * d.y, uvw_first.z + t_min.x * d.z);
  if ( surface_type == 1 ) return float2_t (uvw_first.y + t_max.x * d.y, uvw_first.z + t_max.x * d.z);
  if ( surface_type == 2 ) return float2_t (uvw_first.x + t_min.y * d.x, uvw_first.z + t_min.y * d.z);
  if ( surface_type == 3 ) return float2_t (uvw_first.x + t_max.y * d.x, uvw_first.z + t_max.y * d.z);
  if ( surface_type == 4 ) return float2_t (uvw_first.x + t_min.z * d.x, uvw_first.y + t_min.z * d.y);
  if ( surface_type == 5 ) return float2_t (uvw_first.x + t_max.z * d.x, uvw_first.y + t_max.z * d.y);

  // should not happen -> fallback
  return float2_t(0.0f, 0.0f);
  
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float2 uvw_to_uv ( float3 const& uvw, unsigned surface_type )
{
  if ( surface_type == 0 || surface_type == 1 ) return float2_t (uvw.y, uvw.z);
  if ( surface_type == 2 || surface_type == 3 ) return float2_t (uvw.x, uvw.z);
  if ( surface_type == 4 || surface_type == 5 ) return float2_t (uvw.x, uvw.y);

  // should not happen -> fallback
  return float2_t(0.0f, 0.0f);
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
void write_back_intersection ( int&           intersections, 
                               unsigned&      nfragments,
                               int&           start_index,
                               uint4*         indexlist, 
                               unsigned*      allocation_grid, 
                               unsigned       pagesize, 
                               int2 const&    tilesize, 
                               int2 const&    tileid, 
                               int            fragindex_in, 
                               int            fragindex_out, 
                               unsigned       surface_data_id,
                               float2 const&  uv_guess,
                               float          fdepth )
{
  // write into slot of frontface fragment
  if ( intersections == 0 ) { 
    indexlist[fragindex_in]   = uint4_t(indexlist[fragindex_in].x,  pack_uv(uv_guess), indexlist[fragindex_in].z,   floatBitsToInt(fdepth));   
  }

  // write into slot of backface fragment          
  if ( intersections == 1 ) { 
    indexlist[fragindex_out]  = uint4_t(indexlist[fragindex_out].x, pack_uv(uv_guess), indexlist[fragindex_out].z,  floatBitsToInt(fdepth));   
  }

  // allocate new memory for fragment
  if ( intersections > 1 ) 
  {
    // increase number of "fragments" that is traversed later
    ++nfragments;

    if ( (start_index + 1) % pagesize == 0 ) // page full -> allocate new page
    {
      int chunksize               = tilesize.x * tilesize.y * pagesize;
      int new_start_index         = atomicAdd ( allocation_grid + tileid.y * tilesize.x + tileid.x, chunksize );
      indexlist[new_start_index]  = uint4_t(start_index, pack_uv(uv_guess), surface_data_id, floatBitsToInt(fdepth));   
      start_index                 = new_start_index;
    } else { // write into "old" page
      indexlist[start_index + 1]  = uint4_t(start_index, pack_uv(uv_guess), surface_data_id, floatBitsToInt(fdepth));   
      ++start_index;
    }
  }

  // increment number of total intersections for this face
  ++intersections;
}


///////////////////////////////////////////////////////////////////////////////
inline __device__
bool change_of_partial_derivative ( oriented_point const& current_sample, 
                                    oriented_point const& last_sample, 
                                    float3 const&         ray_direction, 
                                    unsigned              surface_type,
                                    float&                t_extrema ) 
{
  // transform ray to domain
#if 0
  float3 s0_duvw_dt = normalize ( transform_ray_to_domain_space ( ray_direction, last_sample.du,    last_sample.dv,    last_sample.dw ) );
  float3 s1_duvw_dt = normalize ( transform_ray_to_domain_space ( ray_direction, current_sample.du, current_sample.dv, current_sample.dw ) );
#else
  float3 s0_duvw_dt = transform_ray_to_domain_space ( ray_direction, last_sample.du,    last_sample.dv,    last_sample.dw );
  float3 s1_duvw_dt = transform_ray_to_domain_space ( ray_direction, current_sample.du, current_sample.dv, current_sample.dw );
#endif

  float duvw0_dt[3] = {s0_duvw_dt.x, s0_duvw_dt.y, s0_duvw_dt.z};
  float duvw1_dt[3] = {s1_duvw_dt.x, s1_duvw_dt.y, s1_duvw_dt.z};

  float uvw0[3] = {last_sample.uvw.x,    last_sample.uvw.y,    last_sample.uvw.z};
  float uvw1[3] = {current_sample.uvw.x, current_sample.uvw.y, current_sample.uvw.z};

  // take distance to plane u and partial derivative du/dt for cubic 
  float plane  = (surface_type % 2 > 0) ? 1.0f : 0.0f;
  int   coord  = surface_type / 2;

  // compute distance u to plane and partial derivative du/dt
  float u0  = uvw0[coord] - plane;
  float u1  = uvw1[coord] - plane;
  float du0 = duvw0_dt[coord];
  float du1 = duvw1_dt[coord];

  // cubic interpolation : u(t) = at^3 + bt^2 + ct + d
  // and its derivative : u'(t) = 3at^2 + 2bt + c
  float a = 2*u0 - 2*u1 + du1 + du0;
  float b = 3*u1 - 3*u0 - 2*du0 - du1;
  float c = du0;
  float d = u0;

  // solve u'(t)
  float p = (2*b) / (3*a);
  float q = c / (3*a);

  float e = (p*p)/4.0f - q;
  
  if ( e < 0 ) { // no extremum between samples
    return false;
  } else {
    // compute approximative extrema
    float t_ext0 = -(p/2) - sqrt(e);
    float t_ext1 = -(p/2) + sqrt(e);
    
    // compute values at extrema 
    float u_ext0 = a * t_ext0 * t_ext0 * t_ext0 + b * t_ext0 * t_ext0 + c * t_ext0 + d;
    float u_ext1 = a * t_ext1 * t_ext1 * t_ext1 + b * t_ext1 * t_ext1 + c * t_ext1 + d;

    // if extrema are on other side of face, there may be an intersection
    if ( t_ext0 > 0.0f && t_ext0 < 1.0f && u_ext0 * u0 < 0.0f ) 
    {
      t_extrema = t_ext0;
      return true;
    } else {
      if ( t_ext1 > 0.0f && t_ext1 < 1.0f && u_ext1 * u0 < 0.0f ) 
      {
        t_extrema = t_ext1;
        return true;
      } else {
        return false;
      }
    }
  }
}



///////////////////////////////////////////////////////////////////////////////
__device__ inline
void compute_face_intersections_per_domain_intersection ( int&          start_index, 
                                                          unsigned&     nfragments,
                                                          unsigned&     nface_intersections,
                                                          unsigned&     nface_intersection_tests,
                                                          unsigned&     npoint_evaluations,
                                                          int2 const&   screen_coords,
                                                          int2 const&   screen_resolution,
                                                          float         surface_transparency, 
                                                          float         fixed_newton_epsilon,
                                                          unsigned      max_newton_iterations,
                                                          bool          detect_implicit_inflection,
                                                          bool          detect_implicit_extremum,
                                                          float         nearplane,
                                                          float         farplane,
                                                          float         min_sample_distance,
                                                          float         max_sample_distance,
                                                          float         adaptive_sample_scale,
                                                          uint4*        indexlist,
                                                          unsigned*     allocation_grid,
                                                          int2 const&   tileid,
                                                          int2 const&   tilesize,
                                                          unsigned      pagesize,
                                                          uint4 const*  surfacedatabuffer,
                                                          float4 const* surfacepointsbuffer,
                                                          float4 const* volumedatabuffer,
                                                          float4 const* volumepointsbuffer,
                                                          float4 const* modelviewmatrix,
                                                          float4 const* modelviewmatrixinverse,
                                                          float4 const* modelviewprojectionmatrixinverse )
{
  /********************************************************************************
  * allocate variables for ray casting result
  ********************************************************************************/ 
  int    fragindex_in            = start_index;

  /********************************************************************************
  * traverse depth-sorted fragments
  ********************************************************************************/
  while ( fragindex_in != 0 )
  {
    /********************************************************************************
    * retrieve information about fragments, volume and surface
    ********************************************************************************/ 
    uint4    fragindexinfo_in    = indexlist[fragindex_in];
    unsigned fragindex_out       = fragindexinfo_in.x;

    if ( fragindex_out != 0 )
    {
      uint4    fragindexinfo_out   = indexlist [fragindex_out];
      unsigned surface_data_id     = fragindexinfo_in.z;

      /********************************************************************************
      * unpack data for intersection
      ********************************************************************************/ 
      unsigned volume_data_id      = surfacedatabuffer[surface_data_id].z;
      unsigned volume_points_id    = floatBitsToInt(volumedatabuffer[volume_data_id].x);

      uint3    volume_order        = uint3_t ( floatBitsToInt( volumedatabuffer[volume_data_id+1].x),
                                               floatBitsToInt( volumedatabuffer[volume_data_id+1].y),
                                               floatBitsToInt( volumedatabuffer[volume_data_id+1].z) );

      uint4    surface_info        = surfacedatabuffer[surface_data_id + 1];
      uint4    surface_order_info  = surfacedatabuffer[surface_data_id + 2];
      unsigned surface_type        = surface_order_info.w;

      float2   uv_in               = unpack_uv ( fragindexinfo_in.y );
      float2   uv_out              = unpack_uv ( fragindexinfo_out.y );
                                 
      uv_in                        = clamp ( uv_in,  float2_t(0.0f, 0.0f), float2_t(1.0f, 1.0f));
      uv_out                       = clamp ( uv_out, float2_t(0.0f, 0.0f), float2_t(1.0f, 1.0f));
                                 
      float3 uvw_in                = compute_uvw_from_uv ( surfacedatabuffer, surface_data_id, uv_in  );
      float3 uvw_out               = compute_uvw_from_uv ( surfacedatabuffer, surface_data_id, uv_out );
                                 
      uint2   order                = uint2_t ( surface_order_info.y, surface_order_info.z );
      unsigned surface_points_id   = surface_info.y;

      /********************************************************************************
      * assume NO intersection
      ********************************************************************************/
      indexlist[fragindex_in]   = uint4_t(fragindexinfo_in.x,  pack_uv(float2_t(-1.0, -1.0)), fragindexinfo_in.z,  fragindexinfo_in.w);   
      indexlist[fragindex_out]  = uint4_t(fragindexinfo_out.x, pack_uv(float2_t(-1.0, -1.0)), fragindexinfo_out.z, fragindexinfo_out.w);

      /********************************************************************************
      * ray setup
      ********************************************************************************/ 
      float4 ray_entry, ray_exit, ray_entry_max_error;
      float3 n1, n2;
      float  d1, d2;
    
      screen_to_object_coordinates  ( screen_coords, screen_resolution, intBitsToFloat ( fragindexinfo_in.w  ), modelviewprojectionmatrixinverse, &ray_entry);
      screen_to_object_coordinates  ( screen_coords, screen_resolution, intBitsToFloat ( fragindexinfo_out.w ), modelviewprojectionmatrixinverse, &ray_exit);
      screen_to_object_coordinates  ( screen_coords + int2_t(1,1), screen_resolution, intBitsToFloat ( fragindexinfo_in.w ), modelviewprojectionmatrixinverse, &ray_entry_max_error);

      float pixel_error_object_space = length(ray_entry - ray_entry_max_error);

      point_to_plane_intersection   ( ray_entry, modelviewmatrixinverse, &n1, &n2, &d1, &d2 );

      float4 ray_origin         = modelviewmatrixinverse * float4_t(0.0f, 0.0f, 0.0f, 1.0f);
      float4 ray_direction      = ray_exit - ray_origin;
      ray_direction.w           = 0.0f;
      ray_direction             = normalize ( ray_direction );

      // avoid stall by thresholding a minimum sampling distance
      float distance_entry_to_exit  = length ( ray_exit - ray_entry );
      float sample_distance         = max ( MINIMAL_SAMPLING_DISTANCE, distance_entry_to_exit);
      int max_samples               = max ( 2, int(ceil(distance_entry_to_exit/pixel_error_object_space)));

      /********************************************************************************
      * allocate variables necessary for sampling
      ********************************************************************************/ 
      bool is_in_sample      = true;
      bool is_out_sample     = false;
      int intersections      = 0;
      int samples            = 0;

#if INTERSECT_THIN_CONVEX_HULLS_ONCE
      /********************************************************************************
      * for very thin convex hulls -> do single intersection 
      ********************************************************************************/ 
      //bool thin_convex_hull = sample_distance_in_domain_space < min_sample_distance;
      bool thin_convex_hull = distance_entry_to_exit < 2.0f * pixel_error_object_space;//INTERSECT_THIN_CONVEX_HULLS_EPSILON_OBJECT_SPACE;

      if ( thin_convex_hull ) 
      {
        float4 p, du, dv;
        unsigned iterations = 0;
        float2 uv_guess = (uv_in + uv_out)/2.0f;

        ++nface_intersection_tests;
        if ( newton_surface ( surfacepointsbuffer, surface_points_id, uv_guess, fixed_newton_epsilon, max_newton_iterations, order, n1, n2, d1, d2, p, du, dv, iterations) )
        {
          ++nface_intersections;
          float fdepth = compute_depth_from_object_coordinates ( modelviewmatrix, p, nearplane, farplane ); 
          write_back_intersection ( intersections, nfragments, start_index, indexlist, allocation_grid, pagesize, tilesize, tileid, fragindex_in, fragindex_out, surface_data_id, uv_guess, fdepth );
        }

        // just compute this single sample -> do not sample at entry, exit or in between -> simple fallback to single root solution
        is_out_sample = true;
      }
#else
      bool thin_convex_hull = false;
#endif

      /********************************************************************************
      * compute exact sampling distance in uvw space
      ********************************************************************************/ 
      oriented_point in_sample;
      oriented_point out_sample;
      
      if ( !is_out_sample )
      {
        ++npoint_evaluations;
        bool entry_invert_success = newton_volume_unbound ( volumepointsbuffer, 
                                                            volume_points_id, 
                                                            volume_order, 
                                                            uvw_in,
                                                            in_sample.uvw,
                                                            ray_entry, 
                                                            in_sample.p, 
                                                            in_sample.du, 
                                                            in_sample.dv, 
                                                            in_sample.dw, 
                                                            n1, n2, float3_t(ray_direction.x, ray_direction.y, ray_direction.z),
                                                            d1, d2,
                                                            fixed_newton_epsilon, 
                                                            max_newton_iterations );

        ++npoint_evaluations;
        out_sample.uvw = uvw_out;
        bool exit_invert_success = newton_volume_unbound  ( volumepointsbuffer, 
                                                            volume_points_id, 
                                                            volume_order, 
                                                            uvw_out,
                                                            out_sample.uvw,
                                                            ray_exit, 
                                                            out_sample.p, 
                                                            out_sample.du, 
                                                            out_sample.dv, 
                                                            out_sample.dw, 
                                                            n1, n2, float3_t(ray_direction.x, ray_direction.y, ray_direction.z),
                                                            d1, d2,
                                                            fixed_newton_epsilon, 
                                                            max_newton_iterations );
      }

      float sample_distance_in_domain_space = length ( in_sample.uvw - out_sample.uvw );
      float3 uvw_guess           = uvw_in;

      oriented_point current_sample;
      oriented_point last_sample = in_sample;

      /********************************************************************************
      * sample through convex hull and intersect surface
      ********************************************************************************/ 
      int n_resamples         = 0;
      int const max_resamples = 100;

      while ( !is_out_sample )
      {
        if ( is_in_sample )
        {
          current_sample = in_sample;
        } else {
#if ADAPTIVE_SAMPLING
          // adaptive sampling 
          current_sample.p = compute_sampling_position_for_domain_intersection ( last_sample.p, 
                                                                                  ray_direction, 
                                                                                  max_sample_distance,
                                                                                  sample_distance,
                                                                                  pixel_error_object_space,
                                                                                  last_sample.uvw, 
                                                                                  last_sample.du, 
                                                                                  last_sample.dv, 
                                                                                  last_sample.dw,
                                                                                  surface_type );
#else
          current_sample.p = last_sample.p + pixel_error_object_space * ray_direction;
#endif
        }


        uvw_guess  = last_sample.uvw;

        // check if sampling exceeds ray_exit -> if so sample exit and stop sampling
        if ( length ( current_sample.p - ray_entry ) >= distance_entry_to_exit )
        {
          current_sample = out_sample;
          is_out_sample  = true;
        }
        
        /********************************************************************************
        * numerical problem -> critical exit
        ********************************************************************************/ 
        if ( samples++ > max_samples )
        {
          current_sample.p = ray_exit;
          is_out_sample    = true;
        }

        /********************************************************************************
        * inversion of geometric function
        ********************************************************************************/ 
        if ( !is_in_sample && !is_out_sample )
        {
          ++npoint_evaluations;
          float4 sample_pos = current_sample.p;
          bool inversion_success = newton_volume_unbound ( volumepointsbuffer, 
                                                           volume_points_id, 
                                                           volume_order, 
                                                           uvw_guess, 
                                                           current_sample.uvw, 
                                                           sample_pos, 
                                                           current_sample.p, 
                                                           current_sample.du, 
                                                           current_sample.dv, 
                                                           current_sample.dw, 
                                                           n1, n2, float3_t(ray_direction.x, ray_direction.y, ray_direction.z),
                                                           d1, d2,
                                                           fixed_newton_epsilon, 
                                                           max_newton_iterations );
        }

        // stop on unsuccessful inversion
        /*if ( !inversion_success ) {
          break;
        }*/
#if 1
        /********************************************************************************
        * correction of sample position 
        ********************************************************************************/ 
        
        if ( !is_in_sample &&                                             // do not re-position first sample
             (detect_implicit_inflection || detect_implicit_extremum) &&  // do for both: onflections or extrema
             n_resamples++ < max_resamples )                              // limit the number of resamples
        {
          // check if there is is an extrema or inflection between the two samples
          float t_corrected = 0.0f;
          bool sampling_requires_refinement = change_of_partial_derivative ( current_sample, 
                                                                             last_sample, 
                                                                             float3_t(ray_direction.x, ray_direction.y, ray_direction.z), 
                                                                             surface_type, 
                                                                             t_corrected );
          
          // if so, clamp the step length
          float current_sample_distance = length(last_sample.p - current_sample.p);
          t_corrected = clamp ( t_corrected, min(1.0f, pixel_error_object_space / current_sample_distance) , 1.0f );

          if ( sampling_requires_refinement )
          {
            ++npoint_evaluations;

            is_out_sample = false;

            float4 sample_pos = mix (last_sample.p, current_sample.p, t_corrected);
            uvw_guess         = mix (last_sample.uvw, current_sample.uvw, t_corrected);

            bool inversion_success = newton_volume_unbound ( volumepointsbuffer, 
                                                             volume_points_id, 
                                                             volume_order, 
                                                             uvw_guess, 
                                                             current_sample.uvw, 
                                                             sample_pos, 
                                                             current_sample.p, 
                                                             current_sample.du, 
                                                             current_sample.dv, 
                                                             current_sample.dw, 
                                                             n1, n2, float3_t(ray_direction.x, ray_direction.y, ray_direction.z),
                                                             d1, d2,
                                                             fixed_newton_epsilon, 
                                                             max_newton_iterations );

          }
        }
#endif

        // check requirements to trigger an intersection
        bool close_to_boundary                  = min_distance_to_boundary ( current_sample.uvw, float3_t(0.0, 0.0, 0.0), float3_t(1.0, 1.0, 1.0)) <= 2.0f * fixed_newton_epsilon;
        bool in_sample_close_to_boundary        = is_in_sample && close_to_boundary;
        bool out_sample_close_to_boundary       = is_out_sample && close_to_boundary;

        bool domain_intersection                = (surface_type == 0 && last_sample.uvw.x          * current_sample.uvw.x          < 0.0f) ||
                                                  (surface_type == 1 && (last_sample.uvw.x - 1.0f) * (current_sample.uvw.x - 1.0f) < 0.0f) ||
                                                  (surface_type == 2 && last_sample.uvw.y          * current_sample.uvw.y          < 0.0f) ||
                                                  (surface_type == 3 && (last_sample.uvw.y - 1.0f) * (current_sample.uvw.y - 1.0f) < 0.0f) ||
                                                  (surface_type == 4 && last_sample.uvw.z          * current_sample.uvw.z          < 0.0f) ||
                                                  (surface_type == 5 && (last_sample.uvw.z - 1.0f) * (current_sample.uvw.z - 1.0f) < 0.0f);

        /********************************************************************************
        * intersect face, if in or out_samples are close to boundary or isoparametric face is in between two consecutive samples
        ********************************************************************************/ 
        if ( in_sample_close_to_boundary || out_sample_close_to_boundary || domain_intersection ) 
        {
          /********************************************************************************
          * compute uv guess
          ********************************************************************************/ 
          float2 uv_guess;
          if ( domain_intersection && !is_in_sample ) uv_guess = interpolate_uv_guess ( last_sample.uvw, current_sample.uvw, surface_type, min_sample_distance );
          if ( is_in_sample )                         uv_guess = uv_in;
          if ( is_out_sample )                        uv_guess = uv_out;
          

          float4 p, du, dv;
          unsigned iterations = 0;
          ++nface_intersection_tests;
          bool surface_intersection = newton_surface ( surfacepointsbuffer, 
                                                       surface_points_id, 
                                                       uv_guess, 
                                                       fixed_newton_epsilon, 
                                                       max_newton_iterations, 
                                                       order, 
                                                       n1, n2, d1, d2, p, du, dv, iterations);

          // update intersection uv and depth
          float fdepth                          = compute_depth_from_object_coordinates ( modelviewmatrix, p, nearplane, farplane ); 
        
          /********************************************************************************
          * if successful -> store intersection
          ********************************************************************************/
          if ( surface_intersection ) 
          {
            ++nface_intersections;
            write_back_intersection ( intersections, nfragments, start_index, indexlist, allocation_grid, pagesize, tilesize, tileid, fragindex_in, fragindex_out, surface_data_id, uv_guess, fdepth );
          }
        }

        last_sample   = current_sample; // store current sample as last sample for next iteration
        is_in_sample  = false;          // only for first sample
      }

      // go to next fragment
      fragindex_in = fragindexinfo_out.x;
    } else {
      indexlist[fragindex_in]  = uint4_t(fragindexinfo_in.x, pack_uv(float2_t(-1.0, -1.0)), fragindexinfo_in.z, fragindexinfo_in.w);   
      fragindex_in = 0;
    }
  }
}


#endif

