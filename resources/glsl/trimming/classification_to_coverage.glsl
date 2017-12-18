#ifndef GPUCAST_CLASSIFICATION_TO_COVERAGE
#define GPUCAST_CLASSIFICATION_TO_COVERAGE

/////////////////////////////////////////////////////////////////////////////
float normalized_gradient_to_tex_coord(vec2 normalized_gradient) 
{
  const float pi = 3.14159265359;

  // use positive y or mirrored y due to symmetry
  float sin_a = normalized_gradient.y > 0.0 ? normalized_gradient.y : -normalized_gradient.y;

  // compute angle in degree
  float gradient_angle_rad = asin(sin_a);
  float gradient_tex_coord = gradient_angle_rad / (2.0 * pi);

  // return angle in rad
  return gradient_tex_coord;
}

/////////////////////////////////////////////////////////////////////////////
// input:
//    vec2 uv - to be classified uv-point
//    vec2 duv_dx - partial derivivive at uv
//    vec2 duv_dy - partial derivivive at uv
//    bool uv_covered - is center inside or outside -> inside means center is covered with surface
//    vec2 vec2 p_curve - point on curve at which decision was taken
//    vec2 closest bounds - closest bbox point at which decision was taken
//    sampler2D prefilter - prefilter texture using angle and signed distance
// output:
//    float coverage
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
//    x(closest)       |          |
//         *           |          |
//     /\duvdy  *      |          |
//      \           *  |          |
//   ----\------------ x(p)-------|
//   #####x(uv)-->duvdx|  *       |
//   ##################|    *     |
//   ##################|      *   |
//   ##################|        * |
//   ##################|         *|
//   ##################|          *
/////////////////////////////////////////////////////////////////////////////

float
classification_to_coverage(vec2 uv, 
                           vec2 duvdx, 
                           vec2 duvdy, 
                           bool uv_covered, 
                           vec2 p_curve, 
                           vec2 closest_bounds, 
                           sampler2D prefilter)
{
  // if pixel center too far away from curve, return center coverage
  const float max_pixel_distance = length(duvdx + duvdy)/4.0;

  if (max_pixel_distance < length(p_curve-uv)) {
    return float(uv_covered);
  }

  // generate transformation to get from uv into screen coordinates
  //mat2 J = mat2(duvdx, duvdy);
  //mat2 J = mat2(duvdx.x, duvdy.x, duvdx.y, duvdy.y);
  mat2 J = mat2(duvdx.x, duvdx.y, duvdy.x, duvdy.y);

  if (determinant(J) == 0.0) {
    return float(uv_covered);
  }

  mat2 Jinv = inverse(J);

  // transform points to screen coordinates
  vec2 b_pixel_coords = Jinv*closest_bounds; // ok
  vec2 uv_pixel_coords = Jinv*uv; // ok
  vec2 p_pixel_coords = Jinv*p_curve; // ok

  // estimate coverage line through pixel
  vec2 uv_to_curve_point = p_pixel_coords - uv_pixel_coords;
  vec2 tangent = b_pixel_coords - p_pixel_coords;
  vec2 normalized_closest_gradient = normalize(vec2(-tangent.y, tangent.x));

  // transform to angle and signed_distance
  float angle_sin = normalized_closest_gradient.y > 0.0 ? normalized_closest_gradient.y : -normalized_closest_gradient.y;
  float angle_rad = asin(angle_sin);
  float distance_pixel_coords = abs(dot(normalized_closest_gradient, uv_to_curve_point));

  if (!uv_covered) {
    distance_pixel_coords = -distance_pixel_coords;
  } 

  // transform angle and signed distance to texture coordinates of 2D prefilter
  const float sqrt2 = sqrt(2.0);
  const float angle_tex_coords = clamp(normalized_gradient_to_tex_coord(normalized_closest_gradient), 0.0, 1.0);
  const float signed_distance_tex_coords = clamp((distance_pixel_coords + sqrt2 / 2.0) / sqrt2, 0.0, 1.0);

  vec2 prefilter_tex_coords = vec2(angle_tex_coords, signed_distance_tex_coords);

  return texture(prefilter, prefilter_tex_coords).r;
}

#endif