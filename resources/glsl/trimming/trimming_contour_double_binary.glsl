#ifndef GPUCAST_TRIMMING_CONTOUR_DOUBLE_BINARY
#define GPUCAST_TRIMMING_CONTOUR_DOUBLE_BINARY

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/common/conversion.glsl"
#include "resources/glsl/common/constants.glsl"

#include "resources/glsl/math/horner_curve.glsl"
#include "resources/glsl/math/adjoint.glsl.frag"
#include "resources/glsl/math/transfer_function.glsl"

#include "resources/glsl/trimming/binary_search.glsl"
#include "resources/glsl/trimming/bisect_curve.glsl"
#include "resources/glsl/trimming/bisect_contour.glsl"
#include "resources/glsl/trimming/linear_search_contour.glsl"
#include "resources/glsl/trimming/pre_classification.glsl"

////////////////////////////////////////////////////////////////////////////////
// returns classfication using pixel center
////////////////////////////////////////////////////////////////////////////////
bool
trimming_contour_double_binary ( in samplerBuffer domainpartition,
                                 in samplerBuffer contourlist,
                                 in samplerBuffer curvelist,
                                 in samplerBuffer curvedata,
                                 in samplerBuffer pointdata,
                                 in usamplerBuffer preclassification,
                                 in vec2          uv, 
                                 in int           id, 
                                 in int           trim_outer, 
                                 inout int        iters,
                                 in float         tolerance,
                                 in int           max_iterations )
{
  int total_intersections  = 0;
  vec4 baseinfo = texelFetch(domainpartition, id);
  gpucast_count_texel_fetch();

  /////////////////////////////////////////////////////////////////////////////
  // 1. classification : no partition - not trimmed
  /////////////////////////////////////////////////////////////////////////////
  int v_intervals = int(floatBitsToUint(baseinfo.x));
  if (v_intervals == 0) {
    return bool(trim_outer);
  }

  /////////////////////////////////////////////////////////////////////////////
  // 2. trim against outer loop bounds
  /////////////////////////////////////////////////////////////////////////////
  vec4 domaininfo2 = texelFetch(domainpartition, id + 1);
  gpucast_count_texel_fetch();

  // classify against whole domain
  if (uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
      uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2])
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
  // 4. exact classification
  /////////////////////////////////////////////////////////////////////////////
  vec4 vinterval = vec4(0.0);
  bool vinterval_found = binary_search(domainpartition, uv[1], id + 3, v_intervals, vinterval);

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells      = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info = texelFetch(domainpartition, int(celllist_id));
  gpucast_count_texel_fetch();

  vec4 cell           = vec4(0.0);

  bool cellfound = binary_search(domainpartition, uv[0], celllist_id + 1, int(ncells), cell);
  //if (!cellfound) 
  //{
  //  return bool(trim_outer);
  //}

  uvec4 type_ncontours     = intToUInt4 ( floatBitsToUint(cell[2]) );
  total_intersections      = int(type_ncontours.y);
  int overlapping_contours = int(type_ncontours.z);
  int contourlist_id       = int(floatBitsToUint(cell[3]));

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

float length_squared(vec2 a) {
  return a.x*a.x + a.y*a.y;
}

////////////////////////////////////////////////////////////////////////////////
// returns coverage of pixel
////////////////////////////////////////////////////////////////////////////////
float
trimming_contour_double_binary_coverage(in samplerBuffer domainpartition,
                                        in samplerBuffer contourlist,
                                        in samplerBuffer curvelist,
                                        in samplerBuffer curvedata,
                                        in samplerBuffer pointdata,
                                        in usamplerBuffer preclassification,
                                        in sampler2D     prefilter,
                                        in vec2          uv,
                                        in vec2          duvdx,
                                        in vec2          duvdy,
                                        in int           id,
                                        in int           trim_outer,
                                        inout int        iters,
                                        in float         tolerance,
                                        in int           max_iterations,
                                        in int           coverage_estimation_type)
{
  int total_intersections = 0;
  vec4 baseinfo = texelFetch(domainpartition, id);
  gpucast_count_texel_fetch();

  /////////////////////////////////////////////////////////////////////////////
  // 1. classification : no partition - not trimmed
  /////////////////////////////////////////////////////////////////////////////
  int v_intervals = int(floatBitsToUint(baseinfo.x));
  if (v_intervals == 0) {
    return 1.0;
  }

  /////////////////////////////////////////////////////////////////////////////
  // 2. trim against outer loop bounds
  /////////////////////////////////////////////////////////////////////////////
  vec4 domaininfo2 = texelFetch(domainpartition, id + 1);
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
      return float(mod(pre_class, 2) == 1);
    }  
  }

  /////////////////////////////////////////////////////////////////////////////
  // 4. exact classification
  /////////////////////////////////////////////////////////////////////////////
  vec4 vinterval = vec4(0.0);
  bool vinterval_found = binary_search(domainpartition, uv[1], id + 3, v_intervals, vinterval);

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info = texelFetch(domainpartition, int(celllist_id));
  gpucast_count_texel_fetch();

  vec4 cell = vec4(0.0);

  bool cellfound = binary_search(domainpartition, uv[0], celllist_id + 1, int(ncells), cell);

  uvec4 type_ncontours = intToUInt4(floatBitsToUint(cell[2]));
  total_intersections = int(type_ncontours.y);
  int overlapping_contours = int(type_ncontours.z);
  int contourlist_id = int(floatBitsToUint(cell[3]));

  bool found_curve_estimate = false;

  vec2 closest_gradient = vec2(1);
  vec2 closest_point_on_curve = vec2(0);
  float closest_distance = 1.0 / 0.0;

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

  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  // workaround for nasty partial derivative constraints close to preclassified fragments
  /////////////////////////////////////////////////////////////////////////////////////
  if (determinant(J) == 0.0 && classification_base_id != 0)
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
      return float(mod(pre_class, 2) != 0);
    } 
  }
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////


  mat2 Jinv = inverse(J);

  vec2 gradient_pixel_coords = normalize(Jinv*closest_gradient);
  vec2 uv_pixel_coords = Jinv*uv;
  vec2 point_pixel_coords = Jinv*closest_point_on_curve;

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
  };
}


#endif



