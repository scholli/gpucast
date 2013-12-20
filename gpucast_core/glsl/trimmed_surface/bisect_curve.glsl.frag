/*****************************************************************************
 * optimized bisection method
 ****************************************************************************/
void
bisect_curve ( in samplerBuffer curvedata_buffer,
               in vec2          uv,
               in int           index,
               in int           order,
               in bool          horizontally_increasing,
               in float         tmin,
               in float         tmax,
               inout int        intersections,
               inout int        iterations,
               in float         tolerance,
               in int           max_iterations )
{
	float t = 0.0;
  vec4 p  = vec4(0.0);

  int iters = 0;

  for (int i = 0; i < max_iterations; ++i)
  {
    ++iterations;
	  t = (tmax + tmin) / 2.0;
	  evaluateCurve ( curvedata_buffer, index, order, t, p);

    //if ( abs ( uv[1] - p[1] ) < tolerance )
    if ( length ( uv - p.xy ) < tolerance )
    {
      break;
    }

	  if (uv[1] > p[1]) {
	    tmin = t;
	  } else {
	    tmax = t;
	  }

	  if ( (!horizontally_increasing && uv[0] > p[0] && uv[1] > p[1] )||
         ( horizontally_increasing && uv[0] > p[0] && uv[1] < p[1] ) ) 
    {
      break;
	  }

	  if ( (!horizontally_increasing && uv[0] < p[0] && uv[1] < p[1]) ||
         (horizontally_increasing && uv[0] < p[0] && uv[1] > p[1]) )
    {
      ++intersections;
      break;
	  }
	}
}

