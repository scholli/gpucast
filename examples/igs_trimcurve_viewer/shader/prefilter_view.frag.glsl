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

  float intensity = lookup.r;

#if 0
  vec4 vlow = vec4(1.0, 0.0, 1.0, 1.0);
  vec4 low = vec4(0.0, 0.0, 1.0, 1.0);
  vec4 medium = vec4(0.0, 1.0, 0.0, 1.0);
  vec4 high = vec4(1.0, 0.0, 0.0, 1.0);
  vec4 vhigh = vec4(1.0, 1.0, 0.0, 1.0);

  float break1 = 0.2;
  float break2 = 0.5;
  float break3 = 0.8;

  if (intensity <break1)
    outcolor = mix(vlow, low, (intensity - 0.0)/ (break1 - 0.0) );

  if (intensity >= break1 && intensity < break2)
    outcolor = mix(low, medium, (intensity - break1) / (break2 - break1));

  if (intensity >= break2 && intensity < break3)
    outcolor = mix(medium, high, (intensity - break2) / (break3- break2));

  if (intensity >= break3)
    outcolor = mix(high, vhigh, (intensity - break3) / (1.0 - break3));
#else 

  vec4 vlow = vec4(0.0, 0.0, 1.0, 1.0);
  vec4 low = vec4(0.0, 1.0, 0.0, 1.0);
  vec4 medium = vec4(1.0, 1.0, 0.0, 1.0);
  vec4 high = vec4(1.0, 0.0, 0.0, 1.0);

  float break1 = 0.2;
  float break2 = 0.8;

  if (intensity <break1)
    outcolor = mix(vlow, low, (intensity - 0.0) / (break1 - 0.0));

  if (intensity >= break1 && intensity < break2)
    outcolor = mix(low, medium, (intensity - break1) / (break2 - break1));

  if (intensity >= break2)
    outcolor = mix(medium, high, (intensity - break2) / (1.0 - break2));

#endif
}
  
