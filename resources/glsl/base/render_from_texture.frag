/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : render_from_texture.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
uniform sampler2D colorbuffer;

uniform int       width;
uniform int       height;
uniform int       fxaa_mode;

//   1.00 - upper limit (softer)
//   0.75 - default amount of filtering
//   0.50 - lower limit (sharper, less sub-pixel aliasing removal)
//   0.25 - almost off
//   0.00 - completely off
uniform float fxaa_quality_subpix = 1.0;

//   0.333 - too little (faster)
//   0.250 - low quality
//   0.166 - default
//   0.125 - high quality 
//   0.063 - overkill (slower)
uniform float fxaa_edge_threshold = 0.125;

//   0.0833 - upper limit (default, the start of visible unfiltered edges)
//   0.0625 - high quality (faster)
//   0.0312 - visible limit (slower)
uniform float fxaa_threshold_min = 0.0625;

/********************************************************************************
* input
********************************************************************************/
in vec4 frag_texcoord;
in vec4 fxaa_pos;

/********************************************************************************
* output
********************************************************************************/
layout (location=0) out vec4 out_color;

/********************************************************************************
* functions
********************************************************************************/
#include "resources/glsl/common/fxaa_lotthes.glsl"
#include "resources/glsl/common/fxaa_simple.glsl"

/********************************************************************************
* clean pass cleans index image texture
********************************************************************************/
void main(void)
{
  vec2 inverse_resolution = vec2(1.0 / float(width), 1.0 / float(height));

  switch (fxaa_mode)
  {
  case 0:  // No FXAA
    out_color = texture(colorbuffer, frag_texcoord.xy);
    break;  
  case 1:  // SSAA 3.11
    out_color = vec4(FxaaPixelShader(frag_texcoord.xy,
                                colorbuffer,
                                inverse_resolution,
                                fxaa_quality_subpix,
                                fxaa_edge_threshold,
                                fxaa_threshold_min).rgb, 1.0);
    break;
  default:  // Simple FXAA
    out_color = fxaa_simple(colorbuffer, gl_FragCoord.xy, vec2(width, height));
    break;
  }


}
