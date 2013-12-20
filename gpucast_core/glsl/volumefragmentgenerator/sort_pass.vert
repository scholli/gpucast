/********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : volume_multipass/sort_pass.vert
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* attributes
********************************************************************************/
layout (location = 0) in vec4 in_vertex;

/********************************************************************************
* uniforms
********************************************************************************/

/********************************************************************************
* output
********************************************************************************/

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  gl_Position       = in_vertex;
}

