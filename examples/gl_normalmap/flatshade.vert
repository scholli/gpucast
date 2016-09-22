#version 330 compatibility

uniform mat4 modelviewprojectionmatrix;
uniform mat4 modelviewmatrix;
uniform mat4 normalmatrix;

in vec4 vertex;
in vec4 texcoord;
in vec4 normal;

out vec4 frag_position;
out vec4 frag_color;
out vec4 frag_normal;
out vec4 frag_texcoord;

void main()
{
  frag_position = modelviewmatrix * vertex;
  frag_color    = color;
  frag_texcoord = texcoord;
  frag_normal   = normalmatrix * normal;
  
  gl_Position   = modelviewprojectionmatrix * vertex;
}