#ifndef GPUCAST_TRIMMING_LINEAR_SEARCH_CONTOUR
#define GPUCAST_TRIMMING_LINEAR_SEARCH_CONTOUR

bool
linear_search_contour(in samplerBuffer data,
                      in vec2          uv,
                      in int           id,
                      in int           intervals,
                      in bool          uincreasing,
                      inout int        intersections,
                      inout int        curveindex)
{
  int id_min = id;
  int id_max = id + intervals - 1;

  bool found = false;

  for (int i = id_min; i <= id_max; ++i)
  {
    vec4 tmp = texelFetch(data, i);

    // is in v-range
    if (uv[1] >= tmp[0] && uv[1] <= tmp[1])
    {
      if (uv[0] >= tmp[2] && uv[0] <= tmp[3])
      {
        curveindex = id;
        found = true;
      }
      else {
        if (uv[0] < tmp[2])
        {
          ++intersections;
        }
      }
      break;
    }
  }

  return found;
}

#endif