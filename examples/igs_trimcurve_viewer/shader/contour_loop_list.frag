#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_shader_storage_buffer_object : enable

vec4 debug_out = vec4(1);

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/math/transfer_function.glsl"
#include "resources/glsl/trimming/trimming_loop_lists.glsl"

in vec2     uv_coord; 

uniform int trim_index;
uniform int antialiasing;
uniform int show_costs;

uniform sampler2D prefilter_texture;

layout (location = 0) out vec4 outcolor; 

void main(void) 
{ 
  int iterations = 0;
  int sample_rows = 0;
  int sample_cols = 0;

  switch (antialiasing) {

    ///////////////////////////////////////////////////////////////////////////
  case 0: // no anti-aliasing

    bool trimmed = trimming_loop_list(uv_coord, trim_index);

    if (trimmed) {
      outcolor = vec4(0.0);
    }
    else {
      outcolor = debug_out * vec4(1.0);
    }
    break;

    ///////////////////////////////////////////////////////////////////////////
    case 1: // edge-estimate
    case 2: 
    case 3: 

      float coverage = trimming_loop_list_coverage(uv_coord, dFdx(uv_coord), dFdy(uv_coord), prefilter_texture, trim_index, antialiasing);

      outcolor = debug_out * vec4(coverage);
      break;

    ///////////////////////////////////////////////////////////////////////////
    case 4: // 2x2 supersampling
      sample_rows = 2;
      sample_cols = 2;
      break;

      ///////////////////////////////////////////////////////////////////////////
    case 5: // 3x3 supersampling
      sample_rows = 3;
      sample_cols = 3;
      break;

      ///////////////////////////////////////////////////////////////////////////
    case 6: // 4x4 supersampling
      sample_rows = 4;
      sample_cols = 4;
      break;

      ///////////////////////////////////////////////////////////////////////////
    case 7: // 8x8 supersampling
      sample_rows = 8;
      sample_cols = 8;
      break;
  }

  ///////////////////////////////////////////////////////////////////////////
  // supersampling
  ///////////////////////////////////////////////////////////////////////////
  if (sample_rows != 0) {

    float sample_coverage = 0.0;
    vec2 dx = dFdx(uv_coord) / (sample_rows + 1);
    vec2 dy = dFdy(uv_coord) / (sample_cols + 1);
    vec2 uv_base = uv_coord - dFdx(uv_coord) / 2.0 - dFdy(uv_coord) / 2.0;

    for (int c = 1; c <= sample_cols; ++c) {
      for (int r = 1; r <= sample_rows; ++r) {

        sample_coverage += float(!trimming_loop_list(uv_base + r * dx + c * dy, trim_index));
      }
    }
    outcolor = debug_out * vec4(sample_coverage / (sample_rows*sample_cols));
  }

  ///////////////////////////////////////////////////////////////////////////
  // if desired, it shows the costs
  ///////////////////////////////////////////////////////////////////////////
  if (show_costs != 0)
  {
    outcolor = transfer_function(clamp(float(gpucast_texel_fetches) / 64.0, 0.0, 1.0));
  }
}



