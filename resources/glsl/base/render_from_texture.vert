/********************************************************************************
* 
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : render_from_texture.vert
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* attributes
********************************************************************************/
layout (location = 0) in vec4 in_vertex;
layout (location = 1) in vec4 in_texcoord;

/********************************************************************************
* uniforms
********************************************************************************/
uniform float FXAA_SUBPIX_SHIFT = 1.0/4.0;
uniform int width;
uniform int height;

/********************************************************************************
* output
********************************************************************************/
out vec4 frag_texcoord;
out vec4 fxaa_pos;

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  frag_texcoord     = in_texcoord; 

  vec2 frame_ratio  = vec2(1.0/width, 1.0/height);
  fxaa_pos.xy       = in_texcoord.xy;
  fxaa_pos.zw       = in_texcoord.xy - (frame_ratio * (0.5 + FXAA_SUBPIX_SHIFT));

  gl_Position       = in_vertex;
}
