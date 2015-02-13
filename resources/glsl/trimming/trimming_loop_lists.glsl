#ifndef GPUCAST_TRIMMING_LOOP_LIST
#define GPUCAST_TRIMMING_LOOP_LIST

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/common/conversion.glsl"
#include "resources/glsl/trimming/binary_search.glsl"
#include "resources/glsl/trimming/bisect_curve.glsl"

struct bbox_t {
  float umin;
  float umax;
  float vmin;
  float vmax;
};

struct point_t {
  float wx;
  float wy;
  float wz;
  float pad;
};

struct curve_t {
  unsigned int order;
  unsigned int point_index;
  unsigned int uincreasing;
  unsigned int pad;
  bbox_t       bbox;
};

struct loop_t {
  unsigned int nchildren;
  unsigned int child_index;
  unsigned int ncontours;
  unsigned int contour_index;
  bbox_t       bbox;
};

struct contour_t {
  unsigned int ncurves;
  unsigned int curve_index;
  unsigned int uincreasing;
  unsigned int parity_priority;
  bbox_t       bbox;
};


layout(std430) buffer loop_buffer {
  loop_t loops[];
};

layout(std430) buffer contour_buffer {
  contour_t contours[];
};

layout(std430) buffer curve_buffer {
  curve_t curves[];
};

layout(std430) buffer point_buffer {
  vec4 points[];
};


/////////////////////////////////////////////////////////////////////////////////////////////
void
evaluateCurve(in unsigned int index,
              in unsigned int order,
              in float t,
              out vec4 p)
{
  unsigned int deg = order - 1;
  float u = 1.0 - t;

  float bc = 1.0;
  float tn = 1.0;

  p = points[index];
  gpucast_count_texel_fetch();

  p *= u;

  if (order > 2) {
    for (unsigned int i = 1; i <= deg - 1; ++i) {
      tn *= t;
      bc *= (float(deg - i + 1) / float(i));
      p = (p + tn * bc * points[index + i]) * u;
      gpucast_count_texel_fetch();
    }

    p += tn * t * points[index+deg];
    gpucast_count_texel_fetch();
  }
  else {
    /* linear piece*/
    p = mix(points[index], points[index + 1], t);
    gpucast_count_texel_fetch();
    gpucast_count_texel_fetch();
  }

  /* project into euclidian coordinates */
  p[0] = p[0] / p[2];
  p[1] = p[1] / p[2];
}


/////////////////////////////////////////////////////////////////////////////////////////////
void
bisect_curve(in vec2          uv,
             in unsigned int  index,
             in unsigned int  order,
             in bool          horizontally_increasing,
             in float         tmin,
             in float         tmax,
             inout unsigned int intersections,
             in float         tolerance,
             in unsigned int  max_iterations)
{
  // initialize search
  float t = 0.0;
  vec4 p = vec4(0.0);

  // evaluate curve to determine if uv is on left or right of curve
  for (unsigned int i = 0; i < max_iterations; ++i)
  {
    t = (tmax + tmin) / 2.0;
    evaluateCurve(index, order, t, p);

    // stop if point on curve is very close to uv

    //if ( abs ( uv[1] - p[1] ) < tolerance )
    //if (length(uv - p.xy) < tolerance)
    if (abs(uv.x - p.x) + abs(uv.y - p.y) < tolerance)
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


/////////////////////////////////////////////////////////////////////////////////////////////
bool
bisect_contour(in vec2            uv,
               in unsigned int    id,
               in unsigned int    intervals,
               in bool            uincreasing,
               inout unsigned int intersections,
               inout unsigned int curveindex)
{
  unsigned int id_min = id;
  unsigned int id_max = id + intervals - 1;

  bool needs_curve_evealuation = false;

  while (id_min <= id_max)
  {
    unsigned int id = id_min + (id_max - id_min) / unsigned int(2);
    bbox_t curve_bbox = curves[id].bbox;
    gpucast_count_texel_fetch();

    if (uv[1] >= curve_bbox.vmin && uv[1] <= curve_bbox.vmax)
    {
      if (uv[0] >= curve_bbox.umin && uv[0] <= curve_bbox.umax)
      {
        curveindex = id;
        needs_curve_evealuation = true;
        return needs_curve_evealuation;
      }
      else {
        if (uv[0] < curve_bbox.umin)
        {
          ++intersections;
        }
      }
      break;
    }
    else {

      if ((uv[1] < curve_bbox.vmax && uv[0] > curve_bbox.umax && uincreasing) ||
          (uv[1] < curve_bbox.vmin && uv[0] > curve_bbox.umin && uincreasing) ||
          (uv[1] > curve_bbox.vmin && uv[0] > curve_bbox.umax && !uincreasing) ||
          (uv[1] > curve_bbox.vmax && uv[0] > curve_bbox.umin && !uincreasing))
      {
        break;
      }

      if ((uv[1] > curve_bbox.vmin && uv[0] < curve_bbox.umin && uincreasing) ||
          (uv[1] > curve_bbox.vmax && uv[0] < curve_bbox.umax && uincreasing) ||
          (uv[1] < curve_bbox.vmax && uv[0] < curve_bbox.umin && !uincreasing) ||
          (uv[1] < curve_bbox.vmin && uv[0] < curve_bbox.umax && !uincreasing))
      {
        ++intersections;
        break;
      }

      if (uv[1] < curve_bbox.vmin)
      {
        id_max = id - 1;
      }
      else {
        id_min = id + 1;
      }
    }
  }

  return needs_curve_evealuation;
}


/////////////////////////////////////////////////////////////////////////////////////////////
bool is_inside(in bbox_t bbox, in vec2 point) 
{
  return point.x >= bbox.umin && point.x <= bbox.umax &&
         point.y >= bbox.vmin && point.y <= bbox.vmax;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool is_inside(in bbox_t outer, in bbox_t inner)
{
  return is_inside(outer, vec2(inner.umin, inner.vmin)) && 
         is_inside(outer, vec2(inner.umax, inner.vmax));
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool classify_loop(in vec2 uv, in int index)
{
  bbox_t loop_bbox = loops[index].bbox;
  gpucast_count_texel_fetch();

  if (uv.x >= loop_bbox.umin && uv.x <= loop_bbox.umax && uv.y >= loop_bbox.vmin && uv.y <= loop_bbox.vmax)
  {
    unsigned int intersections = 0;

    unsigned int ci = loops[index].contour_index;
    gpucast_count_texel_fetch();

    for (unsigned int i = 0; i != loops[index].ncontours; ++i)
    {
      bbox_t       contour_bbox = contours[ci + i].bbox;
      gpucast_count_texel_fetch();

      // is inside monotonic contour segment
      if (uv[1] >= contour_bbox.vmin && uv[1] <= contour_bbox.vmax)
      {
        if (uv[0] >= contour_bbox.umin && uv[0] <= contour_bbox.umax)
        {
          // curve segment
          unsigned int contour_intersection = 0;
          unsigned int curve_index = 0;

          bool classify_by_curve = bisect_contour(uv,
            contours[ci + i].curve_index,
            contours[ci + i].ncurves,
            contours[ci + i].uincreasing != 0,
            contour_intersection,
            curve_index);

          // classification necessary
          if (classify_by_curve)
          {
            contour_intersection = 0;

            bisect_curve(uv,
              curves[curve_index].point_index,
              curves[curve_index].order,
              curves[curve_index].uincreasing != 0,
              0.0, 1.0,
              contour_intersection,
              0.00001,
              16U);
          }
          intersections += contour_intersection;
        }
        intersections += unsigned int(uv.x < contour_bbox.umin);
      }
    }

    return mod(intersections, 2) == 0;
  }
  else {
    return true;
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
bool
trimming_loop_list (in vec2 uv, in int index)
{
  bool is_trimmed;
  is_trimmed = classify_loop(uv, index);

  for (unsigned int i = 0; i < loops[index].nchildren; ++i) {
    //is_trimmed = is_trimmed && classify_loop(uv, int(loops[index].child_index + i));
    is_trimmed = is_trimmed == classify_loop(uv, int(loops[index].child_index + i));
  }
  return is_trimmed;
}

#endif


