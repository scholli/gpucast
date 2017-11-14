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
// hole-filling
/////////////////////////////////////////////////////////////////////////////
bool examine_potential_hole_fill (inout vec4 color, out float depth)
{
  mat4x2 kernel = mat4x2(vec2(1,1), 
                         vec2(0,1), 
                         vec2(1,0), 
                         vec2(-1,1));

  float pixel_depth = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy), 0).r;
  vec4  pixel_color = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy), 0);

  for (int i = 0; i != 4; ++i) 
  {
    float d0 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy + kernel[i]), 0).r;
    float d1 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy - kernel[i]), 0).r;

    vec4 c0 = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy + ivec2(1,0)), 0);
    vec4 c1 = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy - ivec2(1,0)), 0);

    // todo: threshold adjustment
    if (abs(d0 - d1) < GPUCAST_HOLE_FILLING_THRESHOLD && 
        abs(pixel_depth - d1) > GPUCAST_HOLE_FILLING_THRESHOLD &&
        abs(pixel_depth - d0) > GPUCAST_HOLE_FILLING_THRESHOLD
        )
    {      
      color = (c0 + c1) / 2;
      depth = (d0 + d1) / 2;

      return true;
    }
  }
  return false;
}



/////////////////////////////////////////////////////////////////////////////
// pin-hole-filling
/////////////////////////////////////////////////////////////////////////////
bool examine_color_pinhole (out vec4 fillcolor, float depth_tolerance, float color_tolerance )
{ 
  mat4x2 kernel = mat4x2(vec2(1,1), vec2(1,0), vec2(0,1), vec2(-1,1));
  
  vec3  color = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy), 0).rgb;
  float depth = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy), 0).r;
  
  bool depth_conform = true;
  bool gap_found = false;

  vec4 average_color = vec4(0.0);

  for (int i = 0; i != 4; ++i) 
  {
    float d0 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy + kernel[i]), 0).r;
    float d1 = texelFetch(gpucast_gbuffer_depth, ivec2(gl_FragCoord.xy - kernel[i]), 0).r;

    vec3 c0 = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy + kernel[i]), 0).rgb;
    vec3 c1 = texelFetch(gpucast_gbuffer_color, ivec2(gl_FragCoord.xy - kernel[i]), 0).rgb;

    depth_conform = depth_conform;// && abs((depth - d0) - (d1 - depth)) < depth_tolerance;

    bool same_color = length(c0 - c1) < color_tolerance;
    bool color_gap  = length(c0 - color) > color_tolerance && length(c1 - color) > color_tolerance;

    if (color_gap && same_color) {
      average_color += vec4(c0, 1.0) + vec4(c1, 1.0);
    }

    gap_found = gap_found || (color_gap&& same_color);

    if (!depth_conform) {
      return false;
    }
  }

  if (depth_conform && gap_found) {
    fillcolor = vec4(average_color.rgb/average_color.w, 1.0);
    return true;
  } else {
    return false;
  }
}




/////////////////////////////////////////////////////////////////////////////
void main(void)
{
  vec2 frag_texcoord = gl_FragCoord.xy / vec2(gpucast_resolution);

  float depth      = texture(gpucast_gbuffer_depth, frag_texcoord).r;
  vec4 color       = vec4(0.0);
  
  color = texture(gpucast_gbuffer_color, frag_texcoord);

  float alpha_depth = 1.0;
  bool has_alpha = abuffer_contains_fragments(alpha_depth);
  bool fill_available = false;

#if GPUCAST_HOLE_FILLING

  //////////////////////////
  // fill geometric hole
  //////////////////////////
  vec4  fill_color = vec4(0.0);
  float fill_depth = 0.0;

  fill_available = examine_potential_hole_fill(fill_color, fill_depth);

  if (fill_available) 
  {
    color = fill_color;
    depth = fill_depth - GPUCAST_HOLE_FILLING_THRESHOLD;
  }

  //////////////////////////
  // fill color hole
  //////////////////////////
  #if 1
  bool pixel_is_color_hole = examine_color_pinhole(fill_color, GPUCAST_HOLE_FILLING_THRESHOLD, 0.4);
  if (pixel_is_color_hole) {
     color = fill_color;
  }
  #endif

#endif

#if 1
  vec4 blended_color = vec4(0.0);
  if (has_alpha) {
    float emissivity = 1.0;
    bool saturated = abuffer_blend(color, blended_color, emissivity, depth, 1.0);
    color = vec4(blended_color);
  } 
#endif

  out_color = color;
  //out_color = vec4(depth); 
}
