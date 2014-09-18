/********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : volume_multipass/hull_pass.vert
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#version 420 core
#extension GL_EXT_gpu_shader4 : enable


/********************************************************************************
* attributes
********************************************************************************/
layout (location = 0) in vec4  vertex;
layout (location = 1) in vec4  vertex_attribute;

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4 modelviewprojectionmatrix;
uniform mat4 modelviewmatrix;
uniform mat4 modelviewmatrixinverse;
uniform mat4 normalmatrix;

/********************************************************************************
* output
********************************************************************************/
out vec4        vertex_position;
out vec3        vertex_parameter;
flat out uint   vertex_surface_id;

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  // transform vertex to world for fragment-based ray-generation
  vertex_position                = vertex;

  // forward and rasterize patch and volume information
  vertex_surface_id              = floatBitsToInt(vertex_attribute.w);
  vertex_parameter               = vertex_attribute.xyz;

  // transform vertex to screen for fragment generation
  gl_Position                    = modelviewprojectionmatrix * vertex;
}

