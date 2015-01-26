#ifndef GPUCAST_GLSL_HORNER_CURVE
#define GPUCAST_GLSL_HORNER_CURVE

#include "resources/glsl/common/config.glsl"

/*******************************************************************************
 * Evaluate Curve using modificated horner algorithm in Bernstein basis        *
 *   - points are supposed to be in hyperspace : [wx, wy, w]                   *
 *   - curvedata[index] is the first point of curve                            *
 *   - t is the parameter the curve is to be evaluated for                     *
 ******************************************************************************/
void 
evaluateCurve ( in samplerBuffer data, 
                in int index, 
                in int order, 
                in float t, 
                out vec4 p ) 
{
  int deg = order - 1;
  float u = 1.0 - t;
  
  float bc = 1.0;
  float tn = 1.0;

  p  = texelFetch(data, index);
  gpucast_count_texel_fetch();

  p *= u;

  if (order > 2) {
    for (int i = 1; i <= deg - 1; ++i) {
      tn *= t;
      bc *= (float(deg-i+1) / float(i));
      p = (p + tn * bc * texelFetch(data, index + i)) * u;
      gpucast_count_texel_fetch();
    } 

    p += tn * t * texelFetch(data, index + deg);
    gpucast_count_texel_fetch();
  } else {
    /* linear piece*/
    p = mix(texelFetch(data, index), texelFetch(data, index + 1), t);
    gpucast_count_texel_fetch();
  }
    
  /* project into euclidian coordinates */
  p[0] = p[0]/p[2];
  p[1] = p[1]/p[2];
}

#endif