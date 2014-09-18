/********************************************************************************
* 
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : base.vert
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable

/* attributes*/
layout (location = 0) in vec4 in_vertex;
layout (location = 1) in vec4 in_color;

/* uniforms */
uniform mat4 modelviewprojectionmatrix;

out vec4 color;

void main(void)
{
  gl_Position = modelviewprojectionmatrix * in_vertex;
  color       = in_color;
}

