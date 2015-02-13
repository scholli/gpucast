#extension GL_ARB_separate_shader_objects : enable
#extension GL_NV_gpu_shader5 : enable

in vec2 uv_coord;

uniform sampler2D classification_texture;

uniform int show_costs;
uniform vec2 domain_size;
uniform vec2 domain_min;

layout(location = 0) out vec4 outcolor;

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/math/transfer_function.glsl"

void main(void)
{
  vec4 lookup = texture(classification_texture, uv_coord.xy);

  bool trimmed = lookup.x < 0.0;

  if (show_costs != 0)
  {
    outcolor = transfer_function(clamp(float(gpucast_texel_fetches) / 64.0, 0.0, 1.0));
  }
  else {
    float border = 0.001 / max(abs(lookup.r), 0.00001);
    if (trimmed)
    {
      outcolor = vec4(-lookup.r, 0.0, border, 1.0);
    }
    else {
      outcolor = vec4(0.0, lookup.r, border, 1.0);
    }
  }
}
  
