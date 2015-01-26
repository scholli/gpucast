#ifndef GPUCAST_TRIMMING_CONTOUR_DOUBLE_BINARY
#define GPUCAST_TRIMMING_CONTOUR_DOUBLE_BINARY

#include "resources/glsl/common/conversion.glsl"
#include "resources/glsl/math/horner_curve.glsl"
#include "resources/glsl/trimming/binary_search.glsl"
#include "resources/glsl/trimming/bisect_curve.glsl.frag"
#include "resources/glsl/trimming/bisect_contour.glsl"
#include "resources/glsl/trimming/linear_search_contour.glsl"

bool
trimming_contour_double_binary ( in samplerBuffer domainpartition,
                                 in samplerBuffer contourlist,
                                 in samplerBuffer curvelist,
                                 in samplerBuffer curvedata,
                                 in samplerBuffer pointdata,
                                 in vec2          uv, 
                                 in int           id, 
                                 in int           trim_outer, 
                                 inout int        iters,
                                 in float         tolerance,
                                 in int           max_iterations )
{
  int total_intersections  = 0;
  int v_intervals = int(floatBitsToUint(texelFetch(domainpartition, id).x));

  // if there is no partition in vertical(v) direction -> return
  if ( v_intervals == 0) 
  {
    return false;
  }
  vec4 domaininfo2 = texelFetch(domainpartition, id + 1);

  // classify against whole domain
  if ( uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
       uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2] ) 
  {
    return bool(trim_outer);
  }

  vec4 vinterval = vec4(0.0);
  bool vinterval_found = binary_search(domainpartition, uv[1], id + 2, v_intervals, vinterval);

  //if ( !vinterval_found ) {
  //  return bool(trim_outer);
  //}

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells      = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info = texelFetch(domainpartition, int(celllist_id));
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

    uvec2 ncurves_uincreasing = intToUInt2(floatBitsToUint(contour.x));
    bool contour_uincreasing  = ncurves_uincreasing.y > 0;
    int curves_in_contour     = int(ncurves_uincreasing.x);
    int  curvelist_id         = int(floatBitsToUint(contour.y));
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
      uint pointid    = 0;
      uint curveorder = 0;
      intToUint8_24 ( floatBitsToUint ( curveinfo ), curveorder, pointid );
      bisect_curve ( pointdata, uv, int(pointid), int(curveorder), contour_uincreasing, 0.0f, 1.0f, total_intersections, iters, tolerance, max_iterations );
    }
  }
  
  return ( (mod(total_intersections, 2) == 1) != bool(trim_outer) );
}

#endif



