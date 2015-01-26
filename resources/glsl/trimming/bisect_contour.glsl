#ifndef GPUCAST_TRIMMING_BISECT_CONTOUR
#define GPUCAST_TRIMMING_BISECT_CONTOUR

#include "resources/glsl/common/config.glsl"

bool
bisect_contour(in samplerBuffer data,
               in vec2          uv,
               in int           id,
               in int           intervals,
               in bool          uincreasing,
               inout int        intersections,
               inout int        curveindex )
{
  int id_min = id;
  int id_max = id + intervals - 1;

  vec4 tmp = vec4(0.0);
  bool found = false;

  while ( id_min <= id_max )
  {
    int id = id_min + (id_max - id_min) / int(2);

    tmp = texelFetch(data, id);
    gpucast_count_texel_fetch();

    if ( uv[1] >= tmp[0] && uv[1] <= tmp[1])
    {
      if ( uv[0] >= tmp[2] && uv[0] <= tmp[3] )
      {
        curveindex = id;
        found = true;
      } else { 
        if ( uv[0] < tmp[2] ) 
        {
          ++intersections;
        }
      }
      break;
    } else {
      
      if ( ( uv[1] < tmp[1] && uv[0] > tmp[3] &&  uincreasing ) || 
           ( uv[1] < tmp[0] && uv[0] > tmp[2] &&  uincreasing ) || 
           ( uv[1] > tmp[0] && uv[0] > tmp[3] && !uincreasing ) || 
           ( uv[1] > tmp[1] && uv[0] > tmp[2] && !uincreasing ) )
      {
        break;
      }
      
      if ( ( uv[1] > tmp[0] && uv[0] < tmp[2] &&  uincreasing ) || 
           ( uv[1] > tmp[1] && uv[0] < tmp[3] &&  uincreasing ) ||
           ( uv[1] < tmp[1] && uv[0] < tmp[2] && !uincreasing ) ||
           ( uv[1] < tmp[0] && uv[0] < tmp[3] && !uincreasing ))
      {
        ++intersections;
        break;
      }

      if ( uv[1] < tmp[0] ) 
      {
        id_max = id - 1;
      } else {
        id_min = id + 1;
      }
    }
  }

  return found;
}

#endif