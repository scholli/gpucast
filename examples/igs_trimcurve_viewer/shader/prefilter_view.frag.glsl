#extension GL_ARB_separate_shader_objects : enable
#extension GL_NV_gpu_shader5 : enable

in vec2 uv_coord;

uniform sampler2D prefilter_texture;

uniform int show_costs;
uniform vec2 domain_size;
uniform vec2 domain_min;

layout(location = 0) out vec4 outcolor;

#include "resources/glsl/common/config.glsl"

void main(void)
{
  vec4 lookup = texture(prefilter_texture, uv_coord.xy);

  //float intensity = pow(lookup.r,2);
  float intensity = lookup.r;
  outcolor = vec4(intensity, intensity, intensity, 1.0);
  //outcolor = vec4(uv_coord.xy, 0.0, 1.0);

  if (uv_coord.x > 1)  outcolor = vec4(1.0, 0.0, 0.0, 1.0);
  if (uv_coord.y > 1)  outcolor = vec4(1.0, 0.0, 0.0, 1.0);
}
  
