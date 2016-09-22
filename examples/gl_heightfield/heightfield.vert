
#version 330 compatibility
#extension GL_EXT_gpu_shader4 : enable

uniform mat4 modelviewprojectionmatrix;
uniform mat4 modelviewmatrix;
uniform mat4 normalmatrix;

in vec4 vertex;
in vec4 texcoord;
in vec4 normal;

out vec4 frag_position;
out vec4 frag_normal;
out vec4 frag_texcoord;

void main()
{
  frag_position = modelviewmatrix * vertex;
  frag_texcoord = texcoord;
  frag_normal   = normalmatrix * normal;
  
  gl_Position   = modelviewprojectionmatrix * vertex;
}


