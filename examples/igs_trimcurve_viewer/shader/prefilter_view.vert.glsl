#extension GL_ARB_separate_shader_objects : enable
#extension GL_NV_gpu_shader5 : enable

in vec4 vertex_position;

layout(location = 0) out vec2 uv_coord;

void main(void)
{
  gl_Position = vertex_position;
  uv_coord =  ((vertex_position.xy + vec2(1.0)) / 2.0);
}

