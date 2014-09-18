/********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : raygeneration.glsl.vert                             
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#version 410 core
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* attributes
********************************************************************************/
layout (location = 0) in vec4 in_vertex;
layout (location = 1) in vec4 in_color;
layout (location = 2) in vec4 in_normal;
layout (location = 3) in vec4 in_texcoords;

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4 modelviewprojectionmatrix;

/********************************************************************************
* output
********************************************************************************/
out vec4 frag_position;

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  frag_position = in_vertex;

  // transform vertex to screen for fragment generation
  gl_Position           = modelviewprojectionmatrix * in_vertex;
}

