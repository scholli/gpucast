#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_int64 : enable

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 out_color;


/********************************************************************************
* uniforms
********************************************************************************/
#include "resources/glsl/common/camera_uniforms.glsl"
#include "resources/glsl/trimmed_surface/parametrization_uniforms.glsl"
#include "resources/glsl/abuffer/abuffer_resolve.glsl"

uniform sampler2D gpucast_gbuffer_color;
uniform sampler2D gpucast_gbuffer_depth;

/////////////////////////////////////////////////////////////////////////////
// pin-hole-filling
/////////////////////////////////////////////////////////////////////////////
bool is_pinhole (inout vec4 color, inout float depth)
{
  mat4x2 kernel = mat4x2(vec2(1,1), vec2(1,0), vec2(0,1), vec2(-1,1));
  
  vec3 pixel_color = texture(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy), 0).rgb;
  const float tolerance = 1; // 1 % 

  vec4 average_color = vec4(0.0);
  float average_depth = 0.0;
  int sinks = 0;

  for (int i = 0; i != 4; ++i) 
  {
    vec3 c0 = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy + kernel[i]), 0).rgb;
    vec3 c1 = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy - kernel[i]), 0).rgb;

    float d0 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy + kernel[i]), 0).r;
    float d1 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy - kernel[i]), 0).r;

    average_depth += d0 + d1;
    average_color += vec4(c0, 1.0) + vec4(c1, 1.0);

    bool is_sink = length(c0 - c1) < tolerance * length(c0 - pixel_color) &&
                   length(c0 - c1) < tolerance * length(c1 - pixel_color);

    sinks += int(is_sink);
  }

  if (sinks > 3) {
    color = vec4(average_color.rgb / 8.0, 1.0);
    depth = average_depth / 8.0;
    return true;
  } else {
    return false;
  }
}


/////////////////////////////////////////////////////////////////////////////
// hole-filling
/////////////////////////////////////////////////////////////////////////////
bool examine_potential_hole_fill (inout vec4 color, out float depth)
{
  mat4x2 kernel = mat4x2(vec2(1,1), vec2(0,1), vec2(1,0), vec2(-1,1));
  float pixel_depth = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy), 0).r;

  for (int i = 0; i != 4; ++i) 
  {
    float d0 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy + kernel[i]), 0).r;
    float d1 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy - kernel[i]), 0).r;

    // todo: threshold adjustment
    if (abs(d0 - d1) < GPUCAST_HOLE_FILLING_THRESHOLD && 
        abs(pixel_depth - d1) > GPUCAST_HOLE_FILLING_THRESHOLD &&
        abs(pixel_depth - d0) > GPUCAST_HOLE_FILLING_THRESHOLD
        ) 
    {
      vec4 c0 = texture(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy + kernel[i]), 0);
      vec4 c1 = texture(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy - kernel[i]), 0);

      color = (c0 + c1) / 2;
      depth = (d0 + d1) / 2;

      return true;
    }
  }
  return false;
}

/////////////////////////////////////////////////////////////////////////////
void main(void)
{
  vec2 frag_texcoord = gl_FragCoord.xy / vec2(gpucast_resolution);

  float depth      = texture(gpucast_gbuffer_depth, frag_texcoord).r;
  vec4 color       = vec4(0.0);
  
  color = texture(gpucast_gbuffer_color, frag_texcoord);

  float alpha_depth = 0.0;
  bool has_alpha = abuffer_contains_fragments(alpha_depth);
  bool fill_available = false;

#if GPUCAST_HOLE_FILLING

  vec4  fill_color = vec4(0.0);
  float fill_depth = 0.0;

  fill_available = examine_potential_hole_fill(fill_color, fill_depth);

  if (fill_available ) 
  {
    color = fill_color;
    depth = fill_depth;
  }
#endif

#if 0
  bool pixel_is_pinhole = is_pinhole(fill_color, fill_depth);

  if ( pixel_is_pinhole ) {
    color = fill_color;
    depth = fill_depth;
  }
#endif


#if 1
  vec4 blended_color = vec4(0.0);
  if (has_alpha)
  { 
    float emissivity = 1.0;
    bool saturated = abuffer_blend(color, blended_color, emissivity, depth);
    color = vec4(blended_color);
  } 
#endif

  out_color = color;

}
