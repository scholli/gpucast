/********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : raycast_volume.glsl.vert                             
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* attributes
********************************************************************************/
layout (location = 0) in vec4 vertex;
layout (location = 1) in vec4 vertex_parameter;
layout (location = 2) in vec4 patchinfo;


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
out vec4 fragposition;

flat out int  surface_id;
flat out int  volume_info_id;

flat out int  s_id;
flat out int  t_id;
flat out int  u_id;

out vec3 parameter;

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  // transform vertex to world for fragment-based ray-generation
  fragposition    = vertex;

  // forward and rasterize patch and volume information
  surface_id      = int(vertex_parameter.w);
  volume_info_id  = int(patchinfo.w);

  parameter       = vertex_parameter.xyz;

  s_id            = int(patchinfo.x);
  t_id            = int(patchinfo.y);
  u_id            = int(patchinfo.z);

  // transform vertex to screen for fragment generation
  gl_Position     = modelviewprojectionmatrix * vertex;
}

