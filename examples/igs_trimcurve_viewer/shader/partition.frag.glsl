#version 330 compatibility
#extension GL_ARB_separate_shader_objects : enable 

in vec4 fragcolor;

layout (location = 0) out vec4 color;

void main(void)
{
  color = fragcolor;
}
  
