#version 330 compatibility
#extension GL_EXT_gpu_shader4 : enable

layout (location = 0) in vec4 vertex;   
layout (location = 1) in vec4 texcoord; 

uniform mat4 modelviewprojectionmatrix; 
uniform mat4 modelviewmatrix; 
uniform mat4 normalmatrix; 

out vec4 fragtexcoord;
out vec4 fragposition;

void main(void) 
{ 
  fragtexcoord = texcoord;
  fragposition = modelviewmatrix * vertex;
  gl_Position  = modelviewprojectionmatrix * vertex;
}


