/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : render_from_texture_sao.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* constants
********************************************************************************/
const float SSAO_SCREEN_RADIUS = 40.0;
const float SSAO_DISTANCE = 400.0;
const int SSAO_SAMPLES = 20;

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4      modelviewprojectioninverse;
uniform mat4      modelview;

uniform sampler2D depthbuffer;
uniform sampler2D randomtexture;

uniform int       width;
uniform int       height;
uniform float     nearclip;
uniform float     farclip;


/*******************************************************************************
* input
********************************************************************************/
in vec4 frag_texcoord;

/********************************************************************************
* output
********************************************************************************/
out vec4 out_color;

/********************************************************************************
* functions
********************************************************************************/
#include "resources/glsl/common/ssao.glsl"


/********************************************************************************
* main
********************************************************************************/
void main() 
{
  // in sampler2D depth_buffer, in vec2 uv, in vec2 resolution, in float nearclip, in float farclip, in mat4 mvp_inverse) 

  vec4 ao = compute_ssao(depthbuffer, randomtexture, SSAO_SCREEN_RADIUS, SSAO_DISTANCE, SSAO_SAMPLES, frag_texcoord.xy, vec2(width, height), nearclip, farclip, modelviewprojectioninverse, modelview);

  out_color = vec4(ao.xyz, 1.0);
}