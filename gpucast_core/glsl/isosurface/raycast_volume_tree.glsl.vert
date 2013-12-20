/********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : raycast_volume_tree.glsl.vert                             
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable


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
uniform mat4 modelviewmatrix;
uniform mat4 normalmatrix;

/********************************************************************************
* output
********************************************************************************/
out vec4 fragcolor;
out vec4 fragposition;
out vec4 fragobjectcoordinates;
out vec3 fragnormal;
out vec4 fragtexcoords;

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  fragposition          = modelviewmatrix * in_vertex;
  fragobjectcoordinates = in_vertex;
  
  fragcolor             = in_color;
  fragnormal            = vec3((normalmatrix * normalize(in_normal)).xyz);
  fragtexcoords         = in_texcoords;
  
  // transform vertex to screen for fragment generation
  gl_Position           = modelviewprojectionmatrix * in_vertex;
}

