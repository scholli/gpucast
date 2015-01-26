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

#endif