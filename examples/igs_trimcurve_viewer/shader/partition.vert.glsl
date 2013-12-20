#version 330 compatibility
#extension GL_ARB_separate_shader_objects : enable
    
layout (location = 0) in vec4 vertex;
layout (location = 1) in vec4 color;

uniform mat4 mvp;
   
out vec4 fragcolor;

void main(void)
{
  gl_Position  = mvp * vertex;
  fragcolor = color;
}


