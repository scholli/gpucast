#ifndef GPUCAST_TRIMMING_BISECT_CURVE
#define GPUCAST_TRIMMING_BISECT_CURVE

#include "resources/glsl/math/horner_curve.glsl"

/////////////////////////////////////////////////////////////////////////////
// bisect_curve: fastly detect if uv is left or right on curve
//   - assumptions: 
//     + curve is monotonic in both parameter directions
//     + curvedata_buffer contains control points of curve
/////////////////////////////////////////////////////////////////////////////
void
bisect_curve(in samplerBuffer curvedata_buffer,
             in vec2          uv,
             in int           index,
             in int           order,
             in bool          horizontally_increasing,
             in float         tmin,
             in float         tmax,
             inout int        intersections,
             inout int        iterations,
             in float         tolerance,
             in int           max_iterations)
{
  // initialize search
  float t = 0.0;
  vec4 p = vec4(0.0);
  int iters = 0;

  // evaluate curve to determine if uv is on left or right of curve
  for (int i = 0; i < max_iterations; ++i)
  {
    ++iterations;
    t = (tmax + tmin) / 2.0;
    evaluateCurve(curvedata_buffer, index, order, t, p);

    // stop if point on curve is very close to uv

    //if ( abs ( uv[1] - p[1] ) < tolerance )
    //if (length(uv - p.xy) < tolerance)
    if ( abs(uv.x - p.x) + abs(uv.y - p.y) < tolerance )
    {
      break;
    }

    // classify: no classification of uv is possible -> continue search
    if (uv[1] > p[1]) {
      tmin = t;
    }
    else {
      tmax = t;
    }

    // classify: uv is on left -> stop search
    if ((!horizontally_increasing && uv[0] > p[0] && uv[1] > p[1]) ||
        (horizontally_increasing && uv[0] > p[0] && uv[1] < p[1]))
    {
      break;
    }

    // classify: uv is on right -> stop search
    if ((!horizontally_increasing && uv[0] < p[0] && uv[1] < p[1]) ||
      (horizontally_increasing && uv[0] < p[0] && uv[1] > p[1]))
    {
      ++intersections;
      break;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
vec2 tangent_to_gradient(vec2 tangent) {
  return normalize(vec2(-tangent.y, tangent.x));
}

/////////////////////////////////////////////////////////////////////////////
// bisect_curve: fastly detect if uv is left or right on curve
//   - assumptions: 
//     + curve is monotonic in both parameter directions
//     + curvedata_buffer contains control points of curve
/////////////////////////////////////////////////////////////////////////////
void
bisect_curve_coverage(in samplerBuffer curvedata_buffer,
                      in vec2          uv,
                      in int           index,
                      in int           order,
                      in bool          horizontally_increasing,
                      in float         tmin,
                      in float         tmax,
                      inout int        intersections,
                      inout int        iterations,
                      out vec2         point_on_curve,
                      out vec2         gradient,
                      in float         tolerance,
                      in int           max_iterations,
                      in vec4          curve_bbox)
{
  // initialize search 
  point_on_curve = vec2(0.0);
  gradient = vec2(0.0);
  vec4 remaining_bbox = curve_bbox;

  // evaluate curve to determine if uv is on left or right of curve
  for (int i = 0; i < max_iterations; ++i)
  {
    ++iterations;

    vec4 p = vec4(0.0);
    float t = (tmax + tmin) / 2.0;
    evaluateCurve(curvedata_buffer, index, order, t, p);

    if (!horizontally_increasing) {
      vec2 sekant = remaining_bbox.xw - remaining_bbox.zy;
      gradient = tangent_to_gradient(sekant);
    }
    else {
      vec2 sekant = remaining_bbox.zw - remaining_bbox.xy;
      gradient = tangent_to_gradient(sekant);
    }

    // stop if point on curve is very close to uv

    //if ( abs ( uv[1] - p[1] ) < tolerance )
    //if (length(uv - p.xy) < tolerance)
    if (abs(uv.x - p.x) + abs(uv.y - p.y) < tolerance)
    {
      point_on_curve = p.xy;
      break;
    }
   
    // classify: uv is on right -> no intersection -> stop search
    //   ____________________
    //  | \            xxxxxx|
    //  |   ---____    xxxxxx|
    //  |          \   xxxxxx|
    //  |           --o      |
    //  |                --_ |
    //  | __________________\|
    if (!horizontally_increasing && uv[0] > p[0] && uv[1] > p[1]) {
      vec2 sekant = remaining_bbox.xw - remaining_bbox.zy;
      //gradient = tangent_to_gradient(sekant);
      point_on_curve = p.xy;
      break;
    }

    // classify: uv is on right -> no intersection -> stop search
    //  __________________ 
    // |                 /|
    // |                / |
    // |          __----  |
    // |  ____---oxxxxxxxx|
    // | /        xxxxxxxx|
    // |__________xxxxxxxx|
    if (horizontally_increasing && uv[0] > p[0] && uv[1] < p[1]) {
      vec2 sekant = remaining_bbox.zw - remaining_bbox.xy;
      //gradient = tangent_to_gradient(sekant);
      point_on_curve = p.xy;
      break;
    }

    // classify: uv is on left  -> intersection -> stop search
    //  __________________ 
    // |xxxxxxxxx        /|
    // |xxxxxxxxx       / |
    // |xxxxxxxxx __----  |
    // |  ____---o        |
    // | /                |
    // |__________________|
    if (horizontally_increasing && uv[0] < p[0] && uv[1] > p[1]) {
      ++intersections;
      vec2 sekant = remaining_bbox.zw - remaining_bbox.xy;
      //gradient = tangent_to_gradient(sekant);
      point_on_curve = p.xy;
      break;
    }

    // classify: uv is on left  -> intersection -> stop search
    //   __________________
    //  | \                |
    //  |   ---__          |
    //  |         \        |
    //  |xxxxxxxxxxo__     |
    //  |xxxxxxxxxx   --__ |
    //  |xxxxxxxxxx_______\|  
    if (!horizontally_increasing && uv[0] < p[0] && uv[1] < p[1]) {
      ++intersections;
      vec2 sekant = remaining_bbox.xw - remaining_bbox.zy;
      //gradient = tangent_to_gradient(sekant);
      point_on_curve = p.xy;
      break;
    }

    // classify: no classification of uv is possible -> continue search
    if (uv[1] > p[1]) {
      if (horizontally_increasing) { // keep upper/right
        remaining_bbox = vec4(p.xy, remaining_bbox.zw);
      }
      else { // keep upper/left
        remaining_bbox = vec4(remaining_bbox.x, p.y, p.x, remaining_bbox.w);
      }
      tmin = t;
    }
    else {
      if (horizontally_increasing) { // keep lower/left
        remaining_bbox = vec4(remaining_bbox.xy, p.xy);
      }
      else { // keep lower/right
        remaining_bbox = vec4(p.x, remaining_bbox.y, remaining_bbox.z, p.y);
      }
      tmax = t;
    }

  }
}

#endif