#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable
     
vec4 debug_out = vec4(1.0);

in vec2     uv_coord; 
in vec2     uv_normalized;
     
uniform int trimid;
uniform int antialiasing;
uniform int show_costs;
uniform int pixelsize;

uniform vec2 domain_size;
uniform vec2 domain_min;
uniform vec2 view_size;
uniform vec2 view_min;
uniform float domain_zoom;

uniform samplerBuffer sampler_partition;
uniform samplerBuffer sampler_contourlist;
uniform samplerBuffer sampler_curvelist;
uniform samplerBuffer sampler_curvedata;
uniform samplerBuffer sampler_pointdata;
uniform usamplerBuffer sampler_preclass;

uniform sampler2D     prefilter_texture;

layout (location = 0) out vec4 outcolor; 

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/math/transfer_function.glsl"
#include "resources/glsl/trimming/trimming_contour_kd.glsl"

void main(void) 
{ 
  vec2 uv;
  if ( pixelsize != 1 ) {
    uv = mod(pixelsize*uv_coord, domain_size);
  } else {
    uv = uv_coord;
  }

  vec2 duvdx = dFdx(uv);
  vec2 duvdy = dFdy(uv);

  const float epsilon = 0.00001;
  const int max_iterations = 16;
  const int trim_outer = 1;

  int iterations = 0;
  vec4 debug = vec4 ( 0.0 );
  int sample_rows = 0;
  int sample_cols = 0;

  switch ( antialiasing ) {
  
    ///////////////////////////////////////////////////////////////////////////
    case 0: // no anti-aliasing
      bool trimmed = trimming_contour_kd(sampler_partition,
                                         sampler_contourlist,
                                         sampler_curvelist,
                                         sampler_curvedata,
                                         sampler_pointdata,
                                         sampler_preclass,
                                         uv, 
                                         trimid,
                                         trim_outer,
                                         iterations,
                                         epsilon,
                                         max_iterations);
      outcolor = debug_out * vec4(float(!trimmed));
      break;

      ///////////////////////////////////////////////////////////////////////////
      case 1: // coverage estimate
      case 2: 
      case 3: 
        float coverage = trimming_contour_kd_coverage(sampler_partition,
                                                      sampler_contourlist,
                                                      sampler_curvelist,
                                                      sampler_curvedata,
                                                      sampler_pointdata,
                                                      sampler_preclass,
                                                      prefilter_texture,
                                                      uv,
                                                      duvdx,
                                                      duvdy,
                                                      trimid,
                                                      trim_outer,
                                                      iterations,
                                                      epsilon,
                                                      max_iterations,
                                                      antialiasing);
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
  };
  
  // supersampling
  if (sample_rows != 0) {

    float sample_coverage = 0.0;
    vec2 dx = duvdx / (sample_rows + 1);
    vec2 dy = duvdy / (sample_cols + 1);
    vec2 uv_base = uv - duvdx / 2.0 -duvdy / 2.0;

    for (int c = 1; c <= sample_cols; ++c) {
      for (int r = 1; r <= sample_rows; ++r) {
        
        sample_coverage += float(!trimming_contour_kd(sampler_partition,
          sampler_contourlist, sampler_curvelist, sampler_curvedata, sampler_pointdata, sampler_preclass,
          uv_base + r * dx + c * dy,
          trimid, trim_outer, iterations, epsilon, max_iterations));
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



