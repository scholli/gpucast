#ifndef GPUCAST_TRIMMING_BISECT_CONTOUR
#define GPUCAST_TRIMMING_BISECT_CONTOUR

#include "resources/glsl/common/config.glsl"

///////////////////////////////////////////////////////////////////////////////
// binary classification
///////////////////////////////////////////////////////////////////////////////
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

    tmp = texelFetch(data, id); // [vmin, vmax , umin, umax]
    gpucast_count_texel_fetch();

    if ( uv[1] >= tmp[0] && uv[1] <= tmp[1]) // point is in v-interval
    {
      if ( uv[0] >= tmp[2] && uv[0] <= tmp[3])  // point is in curve bbox
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


///////////////////////////////////////////////////////////////////////////////
// coverage classification
///////////////////////////////////////////////////////////////////////////////
bool
bisect_contour_coverage(in samplerBuffer data,
                        in vec2          uv,
                        in int           id,
                        in int           intervals,
                        in bool          uincreasing,
                        inout int        intersections,
                        inout int        curveindex,
                        inout vec4       bbox,
                        out vec2         classification_point,
                        out vec2         classification_gradient)
{
  int id_min = id;
  int id_max = id + intervals - 1;

  vec4 curve_bbox = vec4(0.0);
  bool found = false;
  vec4 remaining_bbox = bbox;
  classification_point = vec2(-100000);

  while (id_min <= id_max)
  {
    int id = id_min + (id_max - id_min) / int(2);

    curve_bbox = texelFetch(data, id).zxwy; // [umin, vmin, umax, vmax] [vmin, vmax, umin, umax]
    gpucast_count_texel_fetch();

    if (uv[1] >= curve_bbox[1] && uv[1] <= curve_bbox[3]) // point is in v-interval
    {
      if (uv[0] >= curve_bbox[0] && uv[0] <= curve_bbox[2])  // point is in curve bbox
      {
        bbox = curve_bbox; // return bbox of curve [umin, vmin, umax, vmax] 
        curveindex = id;
        found = true;
        //debug_out = vec4(1.0, 1.0, 1.0, 1.0);
      }
      else {
        //debug_out = vec4(1.0, 1.0, 0.0, 1.0);
        //debug_out = vec4(1.0, 0.0, 0.0, 1.0);
        // is in curve's v-interval, left of bbox -> found intersection
        if (uv[0] <= curve_bbox[0]) {
          //  __________________ 
          // |                _/|
          // |          ____ /  |
          // |xxxxxxxxx|    |   |
          // |xxxxxxxxx|____|   |
          // | ____----         |
          // |/_________________|
#if 1
          classification_point = vec2((curve_bbox[0] + curve_bbox[2]) / 2.0, (curve_bbox[1] + curve_bbox[3]) / 2.0); // use center of curve bbox            
#else
          float alpha = curve_bbox[3] - uv[1] / (curve_bbox[3] - curve_bbox[1]);
          classification_point = vec2(curve_bbox[0] + (curve_bbox[2]-curve_bbox[0])*alpha, curve_bbox[1] + (curve_bbox[3]-curve_bbox[1])*alpha);
#endif
          ++intersections;
        }
        else { // is in curve's v-interval, but right -> no intersection
          //  __________________ 
          // |                _/|
          // |          ____ /  |
          // |         |    |xxx|
          // |         |____|xxx|
          // | ____----         |
          // |/_________________|
#if 1
          classification_point = vec2((curve_bbox[0] + curve_bbox[2]) / 2.0, (curve_bbox[1] + curve_bbox[3]) / 2.0); // use center of curve bbox
#else
          float alpha = curve_bbox[3] - uv[1] / (curve_bbox[3] - curve_bbox[1]);
          classification_point = vec2(curve_bbox[0] + (curve_bbox[2] - curve_bbox[0])*alpha, curve_bbox[1] + (curve_bbox[3] - curve_bbox[1])*alpha);
#endif
        }

        if (uincreasing) {
          vec2 sekant = remaining_bbox.zw - remaining_bbox.xy;
          classification_gradient = normalize(vec2(-sekant.y, sekant.x));
        }
        else {
          vec2 sekant = remaining_bbox.xw - remaining_bbox.zy;
          classification_gradient = normalize(vec2(-sekant.y, sekant.x));
        }
      }
      break;
    }
    else {
      //debug_out = vec4(1.0, 1.0, 0.0, 1.0);
      // classify: uv is on right -> no intersection -> stop search
      //  __________________ 
      // |                 /|
      // |                / |
      // |          __----  |
      // |  ____---o        |
      // | /        xxxxxxxx|
      // |__________xxxxxxxx|
      if (uv[1] < curve_bbox[1] && uv[0] > curve_bbox[0] && uincreasing ) 
      {
        classification_point = vec2((curve_bbox[0]+curve_bbox[2])/2.0, (curve_bbox[1]+curve_bbox[3])/2.0); // use center of curve bbox
        vec2 sekant = remaining_bbox.zw - remaining_bbox.xy;
        classification_gradient = normalize(vec2(-sekant.y, sekant.x));
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
      if ( uv[1] > curve_bbox[3] && uv[0] > curve_bbox[0] && !uincreasing ) {
        classification_point = vec2((curve_bbox[0] + curve_bbox[2]) / 2.0, (curve_bbox[1] + curve_bbox[3]) / 2.0); // use center of curve bbox
        vec2 sekant = remaining_bbox.xw - remaining_bbox.zy;
        classification_gradient = normalize(vec2(-sekant.y, sekant.x));
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
      if ( uv[1] > curve_bbox[3] && uv[0] < curve_bbox[2] && uincreasing ) {
        classification_point = vec2((curve_bbox[0] + curve_bbox[2]) / 2.0, (curve_bbox[1] + curve_bbox[3]) / 2.0); // use center of curve bbox
        vec2 sekant = remaining_bbox.zw - remaining_bbox.xy;
        classification_gradient = normalize(vec2(-sekant.y, sekant.x));
        ++intersections;
        break;
      }

      // classify: uv is on left  -> intersection -> stop search
      //   __________________
      //  | \                |
      //  |   ---__          |
      //  |         \        |
      //  |          o__     |
      //  |xxxxxxxxxx   --__ |
      //  |xxxxxxxxxx_______\| 
      if ( uv[1] < curve_bbox[1] && uv[0] < curve_bbox[2] && !uincreasing ) {
        classification_point = vec2((curve_bbox[0] + curve_bbox[2]) / 2.0, (curve_bbox[1] + curve_bbox[3]) / 2.0); // use center of curve bbox
        vec2 sekant = remaining_bbox.xw - remaining_bbox.zy;
        classification_gradient = normalize(vec2(-sekant.y, sekant.x));
        ++intersections;
        break;
      }

      // keep lower contour segment
      if (uv[1] < curve_bbox[1])
      {
        if (uincreasing) {
          remaining_bbox = vec4(remaining_bbox.xy, curve_bbox.xy);
        }
        else {
          remaining_bbox = vec4(curve_bbox[2], remaining_bbox[1], remaining_bbox[2], curve_bbox[1]);
        }
        id_max = id - 1;
      }
      else { // keep upper contour segment
        if (uincreasing) {
          remaining_bbox = vec4(curve_bbox.zw, remaining_bbox.zw);
        }
        else {
          remaining_bbox = vec4(remaining_bbox[0], curve_bbox[3], curve_bbox[0], remaining_bbox[3]);
        }
        id_min = id + 1;
      }
    }
  }

  return found;
}


#endif