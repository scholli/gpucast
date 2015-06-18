#extension GL_ARB_separate_shader_objects : enable
    
layout (location = 0) in vec4 vertex;
layout (location = 1) in vec4 color;

uniform mat4 mvp;
   
uniform vec2 domain_min;
uniform vec2 domain_base;
uniform float domain_zoom;

out vec4 fragcolor;

void main(void)
{
  gl_Position = mvp * ((vertex - vec4(domain_min, 0.0, 0.0)) / vec4(domain_zoom, domain_zoom, domain_zoom, 1.0) + vec4(domain_base, 0.0, 0.0));
  fragcolor = color;
}


