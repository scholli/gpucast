#version 330 compatibility

layout (location = 0) in vec4 vertex;   
layout (location = 1) in vec4 texcoord; 
layout (location = 2) in vec4 normal;   

uniform mat4 modelviewprojectionmatrix; 
uniform mat4 modelviewmatrix; 
uniform mat4 normalmatrix; 

out vec4 fragnormal;  
out vec4 fragtexcoord;
out vec4 fragposition;

void main(void) 
{ 
  fragtexcoord = texcoord;
  fragnormal   = normalmatrix * normal;
  fragposition = modelviewmatrix * vertex;

  gl_Position  = modelviewprojectionmatrix * vertex;
}

