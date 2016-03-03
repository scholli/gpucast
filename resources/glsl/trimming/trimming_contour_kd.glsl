#ifndef GPUCAST_TRIMMING_CONTOUR_KD
#define GPUCAST_TRIMMING_CONTOUR_KD

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/common/conversion.glsl"
#include "resources/glsl/common/constants.glsl"

#include "resources/glsl/math/length_squared.glsl"
#include "resources/glsl/math/horner_curve.glsl"
#include "resources/glsl/math/adjoint.glsl.frag"
#include "resources/glsl/math/transfer_function.glsl"

#include "resources/glsl/trimming/binary_search.glsl"
#include "resources/glsl/trimming/bisect_curve.glsl"
#include "resources/glsl/trimming/bisect_contour.glsl"
#include "resources/glsl/trimming/linear_search_contour.glsl"
#include "resources/glsl/trimming/pre_classification.glsl"


bool
kd_search ( in samplerBuffer search_buffer,
            in vec2          search_uv,
            in int           start_index,
            inout vec4       search_result)
{
#if 1
  // access root info
  vec4 kdnode = texelFetch(search_buffer, start_index);
  gpucast_count_texel_fetch();

  bool is_leaf = intToUInt4 ( floatBitsToUint(kdnode[0]) ).x != 0;

  int max_iters = 0;

  while (!is_leaf) {
    uint direction = intToUInt4 ( floatBitsToUint(kdnode[0]) ).z;

#if 0
    float value = 0.0;
    if (direction == 0) {
      value = search_uv[0];
    } else { 
      value = search_uv[1];
    }

    
    uint child_index = 0;
    
    if (value <= kdnode[1]) { 
      child_index = floatBitsToUint(kdnode[2]);
    } else {
      child_index = floatBitsToUint(kdnode[3]);
    }
#else 
    // without if
    float value = float(direction == 0) * search_uv[0] + (1.0 - float(direction == 0)) * search_uv[1];
    uint child_index = (int(value <= kdnode[1])) * floatBitsToUint(kdnode[2]) + (1 - int(value <= kdnode[1])) * floatBitsToUint(kdnode[3]);
#endif

    // go to child
    kdnode = texelFetch(search_buffer, int(child_index));  
    gpucast_count_texel_fetch();
    is_leaf = intToUInt4 ( floatBitsToUint(kdnode[0]) ).x != 0;
  }

  search_result = kdnode;
  return true;
#else
  int max_iters = 0;

  int current_id = start_index;
  
  while(true) {
    vec4 kdnode = texelFetch(search_buffer, current_id);
    bool is_leaf = intToUInt4 ( floatBitsToUint(kdnode[0]) ).x != 0;

    if (is_leaf) {
      search_result = kdnode;
      return true;
    } else {

      uint direction = intToUInt4 ( floatBitsToUint(kdnode[0]) ).z;
      float value = 0.0;
      if (direction == 0) {
        value = search_uv[0];
      } else { 
        value = search_uv[1];
      }

      if (value <= kdnode[1]) { 
        current_id = int(floatBitsToUint(kdnode[2]));
      } else {
        current_id = int(floatBitsToUint(kdnode[3]));
      }
    }

    if (max_iters++ > 30) {
      return false;
    }
  }

#endif
}

////////////////////////////////////////////////////////////////////////////////
// returns classfication using pixel center
////////////////////////////////////////////////////////////////////////////////
bool
trimming_contour_kd ( in samplerBuffer  domainpartition,
                      in samplerBuffer  contourlist,
                      in samplerBuffer  curvelist,
                      in samplerBuffer  curvedata,
                      in samplerBuffer  pointdata,
                      in usamplerBuffer preclassification,
                      in vec2           uv, 
                      in int            id, 
                      in int            trim_outer, 
                      inout int         iters,
                      in float          tolerance,
                      in int            max_iterations )
{
  int total_intersections  = 0;
  vec4 baseinfo = texelFetch(domainpartition, id);
  gpucast_count_texel_fetch();

  int nnodes = int(floatBitsToUint(baseinfo.x));

  /////////////////////////////////////////////////////////////////////////////
  // 1. classification : no partition - not trimmed
  /////////////////////////////////////////////////////////////////////////////
  if ( nnodes <= 0) {
    return bool(trim_outer);
  }

  /////////////////////////////////////////////////////////////////////////////
  // 2. trim against outer loop bounds
  /////////////////////////////////////////////////////////////////////////////
  vec4 domaininfo2  = texelFetch(domainpartition, id + 1);
  gpucast_count_texel_fetch();

  // classify against whole domain
  if ( uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
       uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2] ) 
  {
    return bool(trim_outer);
  }

  /////////////////////////////////////////////////////////////////////////////
  // 3. texture-based pre-classification 
  /////////////////////////////////////////////////////////////////////////////
  int classification_base_id = int(floatBitsToUint(baseinfo.y));
  vec4 domainbounds = texelFetch(domainpartition, id + 2);
  gpucast_count_texel_fetch();

  if (classification_base_id != 0)
  {
    int preclasstex_width  = int(floatBitsToUint(baseinfo.z));
    int preclasstex_height = int(floatBitsToUint(baseinfo.w));    
    int pre_class = pre_classify(preclassification,
                                 classification_base_id,
                                 uv,
                                 domainbounds,
                                 preclasstex_width, 
                                 preclasstex_height);
  
    if (pre_class != 0) {
      return mod(pre_class, 2) == 0;
    }  
  }

  /////////////////////////////////////////////////////////////////////////////
  // 4. magnification - kdtree classification
  /////////////////////////////////////////////////////////////////////////////
  vec4 kdnode = vec4(0.0);
  float tmp1;
  bool kdnode_found = kd_search(domainpartition, uv, id + 3, kdnode);

  uvec4 kdnode_info        = intToUInt4 ( floatBitsToUint(kdnode[0]) );

  total_intersections      = int(kdnode_info.y);
  int overlapping_contours = int(kdnode_info.w);
  int contourlist_id       = int(floatBitsToUint(kdnode[1]));

  for ( int i = 0; i < overlapping_contours; ++i )
  {
    vec2 contour    = texelFetch ( contourlist, contourlist_id + i ).xy;
    gpucast_count_texel_fetch();

    uvec2 ncurves_uincreasing = intToUInt2(floatBitsToUint(contour.x));
    bool contour_uincreasing  = ncurves_uincreasing.y > 0;
    int curves_in_contour     = int(ncurves_uincreasing.x);
    int curvelist_id          = int(floatBitsToUint(contour.y));
    vec4 curve_bbox           = vec4(0);
    int curveid               = 0;

    bool process_curve = bisect_contour(curvelist,
                                        uv, 
                                        curvelist_id, 
                                        curves_in_contour, 
                                        contour_uincreasing, 
                                        total_intersections,
                                        curveid );

    //bool process_curve = contour_linear_search(curvelist,
    //  uv,
    //  curvelist_id,
    //  curves_in_contour,
    //  contour_uincreasing,
    //  total_intersections,
    //  curveid);

    if ( process_curve ) 
    {
      int iters = 0;
      float curveinfo = texelFetch (curvedata, curveid).x;
      gpucast_count_texel_fetch();

      uint pointid    = 0;
      uint curveorder = 0;
      intToUint8_24 ( floatBitsToUint ( curveinfo ), curveorder, pointid );
      bisect_curve ( pointdata, uv, int(pointid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iters, tolerance, max_iterations );
    }
  }
  
  return ((mod(total_intersections, 2) == 1) != bool(trim_outer));
}

////////////////////////////////////////////////////////////////////////////////
// returns coverage of pixel
////////////////////////////////////////////////////////////////////////////////
float
trimming_contour_kd_coverage(in samplerBuffer  domainpartition,
                             in samplerBuffer  contourlist,
                             in samplerBuffer  curvelist,
                             in samplerBuffer  curvedata,
                             in samplerBuffer  pointdata,
                             in usamplerBuffer preclassification,
                             in sampler2D      prefilter,
                             in vec2           uv,
                             in vec2           duvdx,
                             in vec2           duvdy,
                             in int            id,
                             in int            trim_outer,
                             inout int         iters,
                             in float          tolerance,
                             in int            max_iterations,
                             in int            coverage_estimation_type)
{
  int total_intersections = 0;
  vec4 baseinfo = texelFetch(domainpartition, id);
  gpucast_count_texel_fetch();

  /////////////////////////////////////////////////////////////////////////////
  // 1. classification : no partition - not trimmed
  /////////////////////////////////////////////////////////////////////////////
  if (int(floatBitsToUint(baseinfo.x)) == 0) {
    return 1.0;
  } 

  /////////////////////////////////////////////////////////////////////////////
  // 2. trim against outer loop bounds
  /////////////////////////////////////////////////////////////////////////////
  vec4 domaininfo2  = texelFetch(domainpartition, id + 1);
  gpucast_count_texel_fetch();

  // classify against whole domain
  if (uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
      uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2])
  {
    return float(!bool(trim_outer));
  } 

  /////////////////////////////////////////////////////////////////////////////
  // 3. texture-based pre-classification 
  /////////////////////////////////////////////////////////////////////////////
  int classification_base_id = int(floatBitsToUint(baseinfo.y));
  vec4 domainbounds = texelFetch(domainpartition, id + 2);
  gpucast_count_texel_fetch();

  if (classification_base_id != 0)
  {
    int preclasstex_width  = int(floatBitsToUint(baseinfo.z));
    int preclasstex_height = int(floatBitsToUint(baseinfo.w));    
    int pre_class = pre_classify(preclassification,
                                 classification_base_id,
                                 uv,
                                 domainbounds,
                                 preclasstex_width, 
                                 preclasstex_height);
  
    if (pre_class != 0) {
      float pre_classification_result = float(mod(pre_class, 2) != 0);
      return pre_classification_result;
    } 
  }

  /////////////////////////////////////////////////////////////////////////////
  // 4. magnification - kdtree classification
  /////////////////////////////////////////////////////////////////////////////
  vec4 kdnode = vec4(0.0);
  bool kdnode_found = kd_search(domainpartition, uv, id + 3, kdnode);

  uvec4 kdnode_info        = intToUInt4 ( floatBitsToUint(kdnode[0]) );
  total_intersections      = int(kdnode_info.y);
  int overlapping_contours = int(kdnode_info.w);
  int contourlist_id       = int(floatBitsToUint(kdnode[1]));

  vec2 closest_gradient = vec2(1);
  vec2 closest_point_on_curve = vec2(0);
  float closest_distance = 1.0;

  for (int i = 0; i < overlapping_contours; ++i)
  {
    vec2 gradient = vec2(0);
    vec2 point_on_curve = vec2(0.0);
    vec4 contour = texelFetch(contourlist, contourlist_id + i);
    gpucast_count_texel_fetch();

    uvec2 ncurves_uincreasing = intToUInt2(floatBitsToUint(contour.x));
    bool contour_uincreasing = ncurves_uincreasing.y > 0;
    int curves_in_contour = int(ncurves_uincreasing.x);
    int curvelist_id = int(floatBitsToUint(contour.y));
    int curveid = 0;

    vec4 remaining_bbox = vec4(unpackHalf2x16(floatBitsToUint(contour.z)), unpackHalf2x16(floatBitsToUint(contour.w)));

    bool process_curve = bisect_contour_coverage(curvelist,
      uv,
      curvelist_id,
      curves_in_contour,
      contour_uincreasing,
      total_intersections,
      curveid,
      remaining_bbox,
      point_on_curve,
      gradient);

    if (process_curve)
    {
      int iterations = 0;
      float curveinfo = texelFetch(curvedata, curveid).x;
      gpucast_count_texel_fetch();

      uint pointid = 0;
      uint curveorder = 0;
      intToUint8_24(floatBitsToUint(curveinfo), curveorder, pointid);
      bisect_curve_coverage(pointdata, uv, int(pointid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iterations, point_on_curve, gradient, tolerance, max_iterations, remaining_bbox);

      float distance = length_squared(point_on_curve.xy - uv.xy);
      if (distance < closest_distance) 
      {
        closest_distance = distance;
        closest_point_on_curve = point_on_curve;
        closest_gradient = gradient;
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////
  // coverage estimation
  /////////////////////////////////////////////////////////////////////////////////////
  bool covered = bool((mod(total_intersections, 2) == 1) == bool(trim_outer));
  mat2 J = mat2(duvdx, duvdy);

  if (determinant(J) == 0.0) {
    return float(!covered);
  }

  mat2 Jinv = inverse(J);

  vec2 gradient_pixel_coords = normalize(Jinv*closest_gradient); // ok
  vec2 uv_pixel_coords = Jinv*uv; // ok
  vec2 point_pixel_coords = Jinv*closest_point_on_curve; // ok

  float distance_pixel_coords = abs(dot(gradient_pixel_coords, uv_pixel_coords - point_pixel_coords));
  const float sqrt2 = sqrt(2.0);

  if (!covered) {
    distance_pixel_coords = -distance_pixel_coords;
  }

  const float gradient_angle = gradient_pixel_coords.y > 0.0 ? gradient_pixel_coords.y : -gradient_pixel_coords.y;
  const float normalized_signed_distance = clamp((distance_pixel_coords + sqrt2 / 2.0) / sqrt2, 0.0, 1.0);

  switch (coverage_estimation_type) 
  { 
    case 1: // edge estimation
      return texture2D(prefilter, vec2(gradient_angle, normalized_signed_distance)).r;
    case 2: // curve estimation -> TODO: not implemented yet
      return texture2D(prefilter, vec2(gradient_angle, normalized_signed_distance)).r;
    case 3: // distance estimation
      return min(1.0, distance_pixel_coords + sqrt(2.0) / 2.0);
    default : 
      return 0.0;
  };

}


#endif



