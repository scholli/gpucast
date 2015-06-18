#ifndef GPUCAST_TRIMMING_DOUBLE_BINARY
#define GPUCAST_TRIMMING_DOUBLE_BINARY

#include "resources/glsl/common/config.glsl"

#include "resources/glsl/trimming/binary_search.glsl"
#include "resources/glsl/trimming/bisect_curve.glsl"

///////////////////////////////////////////////////////////////////////////////
bool
trimming_double_binary ( in samplerBuffer partition_buffer,
                         in samplerBuffer cell_buffer, 
                         in samplerBuffer curvelist_buffer, 
                         in samplerBuffer curvedata_buffer, 
                         in vec2    uv, 
                         in int     id, 
                         in int     trim_outer, 
                         inout int  iters,
                         in float   tolerance,
                         in int     max_iterations)
{
  vec4 domaininfo1 = texelFetch( partition_buffer, id );
  gpucast_count_texel_fetch();

  int total_intersections  = 0;
  int v_intervals = int ( floatBitsToUint ( domaininfo1[0] ) );

  // if there is no partition in vertical(v) direction -> return
  if ( v_intervals == 0) 
  {
    return false;
  } 
  
  vec4 domaininfo2 = texelFetch ( partition_buffer, id+1 );
  gpucast_count_texel_fetch();

  // classify against whole domain
  if ( uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
       uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2] ) 
  {
    return bool(trim_outer);
  }

  vec4 vinterval = vec4(0.0, 0.0, 0.0, 0.0);
  bool vinterval_found = binary_search ( partition_buffer, uv[1], id + 2, v_intervals, vinterval );

  if ( !vinterval_found ) {
    return bool(trim_outer);
  }

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells      = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info  = texelFetch(cell_buffer, int(celllist_id));
  gpucast_count_texel_fetch();

  vec4 cell           = vec4(0.0);

  bool cellfound      = binary_search   (cell_buffer, uv[0], celllist_id + 1, int(ncells), cell );
  if (!cellfound) 
  {
    return bool(trim_outer);
  }

  vec4 clist                       = texelFetch(curvelist_buffer, int(floatBitsToUint(cell[3])));
  gpucast_count_texel_fetch();

  total_intersections              = int(floatBitsToUint(cell[2]));
  unsigned int curves_to_intersect = floatBitsToUint(clist[0]);

  for (int i = 1; i <= curves_to_intersect; ++i) 
  {
    vec4 curveinfo = texelFetch ( curvelist_buffer, int(floatBitsToUint(cell[3])) + i );
    gpucast_count_texel_fetch();

    int index = int(floatBitsToUint(curveinfo[0]));
    int order = abs(floatBitsToInt(curveinfo[1]));
    bool horizontally_increasing = floatBitsToInt(curveinfo[1]) > 0;

    bisect_curve(curvedata_buffer, uv, index, order, horizontally_increasing, 0.0, 1.0, total_intersections, iters, tolerance, max_iterations);
  }

  if ( mod(total_intersections, 2) == 1 ) 
  {
    return (!bool(trim_outer));
  } else {
    return bool(trim_outer);
  }
}

///////////////////////////////////////////////////////////////////////////////
float 
trimming_double_binary_coverage(in samplerBuffer partition_buffer,
                                in samplerBuffer cell_buffer,
                                in samplerBuffer curvelist_buffer,
                                in samplerBuffer curvedata_buffer,
                                in sampler2D prefilter,
                                in vec2    uv,
                                in vec2    duvdx,
                                in vec2    duvdy,
                                in int     id,
                                in int     trim_outer,
                                inout int  iters,
                                in float   tolerance,
                                in int     max_iterations)
{
  vec4 domaininfo1 = texelFetch(partition_buffer, id);
  gpucast_count_texel_fetch();

  int total_intersections = 0;
  int v_intervals = int(floatBitsToUint(domaininfo1[0]));

  vec2 closest_point_on_curve = vec2(0);
  vec2 closest_gradient_on_curve = vec2(0);
  vec4 remaining_bbox = vec4(0);

  // if there is no partition in vertical(v) direction -> return
  if (v_intervals == 0) {
    return 0.0;
  }

  vec4 domaininfo2 = texelFetch(partition_buffer, id + 1);
  gpucast_count_texel_fetch();

  // classify against whole domain
  if (uv[0] > domaininfo2[1] || uv[0] < domaininfo2[0] ||
      uv[1] > domaininfo2[3] || uv[1] < domaininfo2[2])
  {
    return float(!bool(trim_outer));
  }

  vec4 vinterval = vec4(0.0, 0.0, 0.0, 0.0);
  bool vinterval_found = binary_search(partition_buffer, uv[1], id + 2, v_intervals, vinterval);

  if (!vinterval_found) {
    return float(!bool(trim_outer));
  }

  remaining_bbox[1] = vinterval[0]; // vmin;
  remaining_bbox[3] = vinterval[1]; // vmax;

  int celllist_id = int(floatBitsToUint(vinterval[2]));
  int ncells = int(floatBitsToUint(vinterval[3]));

  vec4 celllist_info = texelFetch(cell_buffer, int(celllist_id));
  gpucast_count_texel_fetch();

  vec4 cell = vec4(0.0);

  bool cellfound = binary_search(cell_buffer, uv[0], celllist_id + 1, int(ncells), cell);
  if (!cellfound)
  {
    return float(!bool(trim_outer));
  }

  remaining_bbox[0] = cell[0]; // umin;
  remaining_bbox[2] = cell[1]; // umax;

  vec4 clist = texelFetch(curvelist_buffer, int(floatBitsToUint(cell[3])));
  gpucast_count_texel_fetch();

  total_intersections = int(floatBitsToUint(cell[2]));
  unsigned int curves_to_intersect = floatBitsToUint(clist[0]);

  for (int i = 1; i <= curves_to_intersect; ++i)
  {
    vec4 curveinfo = texelFetch(curvelist_buffer, int(floatBitsToUint(cell[3])) + i);
    gpucast_count_texel_fetch();

    int index = int(floatBitsToUint(curveinfo[0]));
    int order = abs(floatBitsToInt(curveinfo[1]));
    bool horizontally_increasing = floatBitsToInt(curveinfo[1]) > 0;

    bisect_curve_coverage(curvedata_buffer, uv, index, order, horizontally_increasing, 0.0, 1.0, total_intersections, iters, closest_point_on_curve, closest_gradient_on_curve, tolerance, max_iterations, remaining_bbox);
  }

  bool is_trimmed = (mod(total_intersections, 2) == 1) ? !bool(trim_outer) : bool(trim_outer);

  // if no curves present -> directly return full or no coverage
  if (curves_to_intersect == 0) {
    return float(!is_trimmed);
  }

  /////////////////////////////////////////////////////////////////////////////
  // coverage estimation
  /////////////////////////////////////////////////////////////////////////////
  mat2 J = mat2(duvdx, duvdy);
  mat2 Jinv = inverse(J);
  //mat2 Jinv = adjoint(J) / determinant(J);

  vec2 gradient_pixel_coords = normalize(Jinv*closest_gradient_on_curve);
  vec2 uv_pixel_coords = Jinv*uv;
  vec2 point_pixel_coords = Jinv*closest_point_on_curve;

  float distance_pixel_coords = abs(dot(gradient_pixel_coords, uv_pixel_coords - point_pixel_coords));

  if (is_trimmed) {
    distance_pixel_coords = -distance_pixel_coords;
  }

  const float sqrt2 = sqrt(2.0);
  const float gradient_angle = gradient_pixel_coords.y > 0.0 ? gradient_pixel_coords.y : -gradient_pixel_coords.y;
  const float normalized_signed_distance = (distance_pixel_coords + sqrt2 / 2.0) / sqrt2;

  return texture2D(prefilter, vec2(gradient_angle, normalized_signed_distance)).r;
}


#endif


