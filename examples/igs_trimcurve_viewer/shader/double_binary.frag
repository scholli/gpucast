#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable

vec4 debug_out = vec4(1);
in vec2     uv_coord; 

uniform int trimid;
uniform int antialiasing;
uniform int show_costs;

uniform samplerBuffer trimdata;
uniform samplerBuffer celldata;
uniform samplerBuffer curvelist;
uniform samplerBuffer curvedata;

uniform sampler1D     transfertexture;
uniform sampler2D     prefilter_texture;

layout (location = 0) out vec4 outcolor; 

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/math/transfer_function.glsl"
#include "resources/glsl/trimming/trimming_double_binary.glsl"

void main(void) 
{ 
  int iterations = 0;
  vec4 debug = vec4(0.0);
  int sample_rows = 0;
  int sample_cols = 0;

  switch (antialiasing) {

    ///////////////////////////////////////////////////////////////////////////
  case 0: // no anti-aliasing
    bool trimmed = trimming_double_binary(trimdata, celldata, curvelist, curvedata, uv_coord, trimid, 1, iterations, 0.00001, 16);

    if (trimmed) {
      outcolor = vec4(0.0);
    }
    else {
      outcolor = debug_out * vec4(1.0);
    }
    break;

    ///////////////////////////////////////////////////////////////////////////
  case 1: // edge-estimate
    float coverage = trimming_double_binary_coverage(trimdata, celldata, curvelist, curvedata,
      prefilter_texture,
      uv_coord,
      dFdx(uv_coord),
      dFdy(uv_coord),
      trimid,
      1,
      iterations,
      0.000001,
      16);
    outcolor = debug_out * vec4(coverage);
    break;

    ///////////////////////////////////////////////////////////////////////////
  case 2: // 2x2 supersampling
    sample_rows = 2;
    sample_cols = 2;
    break;

    ///////////////////////////////////////////////////////////////////////////
  case 3: // 3x3 supersampling
    sample_rows = 3;
    sample_cols = 3;
    break;

    ///////////////////////////////////////////////////////////////////////////
  case 4: // 4x4 supersampling
    sample_rows = 4;
    sample_cols = 4;
    break;

    ///////////////////////////////////////////////////////////////////////////
  case 5: // 8x8 supersampling
    sample_rows = 8;
    sample_cols = 8;
    break;
  };

  // supersampling
  if (sample_rows != 0) {

    float sample_coverage = 0.0;
    vec2 dx = dFdx(uv_coord) / (sample_rows + 1);
    vec2 dy = dFdy(uv_coord) / (sample_cols + 1);
    vec2 uv_base = uv_coord - dFdx(uv_coord) / 2.0 - dFdy(uv_coord) / 2.0;

    for (int c = 1; c <= sample_cols; ++c) {
      for (int r = 1; r <= sample_rows; ++r) {
        sample_coverage += float(!trimming_double_binary(trimdata, celldata, curvelist, curvedata, uv_base + r * dx + c * dy, trimid, 1, iterations, 0.00001, 16));
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



