#extension GL_ARB_separate_shader_objects : enable 
     
in      vec4 vertex_position; 

uniform float border;

uniform int width;
uniform int height;

uniform vec2 domain_size;
uniform vec2 domain_min;
uniform float domain_zoom;
     
layout (location = 0) out vec2 uv_coord; 
     
void main(void) 
{ 
  gl_Position = vertex_position;
  uv_coord = domain_min + ((vertex_position.xy + vec2(1.0)) / 2.0) * (domain_size)* domain_zoom;
}




