/********************************************************************************
* 
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : tesselator.vert                             
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable


/********************************************************************************
* attributes
********************************************************************************/
layout (location = 0) in vec4 vertex;
layout (location = 1) in vec4 color;

/********************************************************************************
* uniforms
********************************************************************************/
uniform mat4          modelviewprojectionmatrix;
uniform mat4          modelviewmatrix;
uniform mat4          modelviewmatrixinverse;
uniform mat4          normalmatrix;

/********************************************************************************
* output
********************************************************************************/
out vec4 fragcolor;
out vec4 fragposition;

/********************************************************************************
* vertex program for raycasting bezier volumes
********************************************************************************/
void main(void)
{
  fragcolor     = color; 
  fragposition  = modelviewmatrix * vertex;

  // transform vertex to screen for fragment generation
  gl_Position   = modelviewprojectionmatrix * vertex;
}

