#ifndef GPUCAST_PRE_CLASSIFICATION_GLSL
#define GPUCAST_PRE_CLASSIFICATION_GLSL      

#include "resources/glsl/common/config.glsl"

int pre_classify(in usamplerBuffer preclassification,
                  in int base_id,
                  in vec2 uv,
                  in vec4 domainsize,
                  in int pre_tex_width, 
                  in int pre_tex_height) 
{
  // domainsize = [umin, umax, vmin, vmax]
  float size_u = domainsize[1] - domainsize[0];
  float size_v = domainsize[3] - domainsize[2];

  int texel_u = int( pre_tex_width  * (uv[0] - domainsize[0]) / size_u);
  int texel_v = int( pre_tex_height * (uv[1] - domainsize[2]) / size_v);

  int preclass_id = base_id + texel_v * pre_tex_width + texel_u;

  uint preclass = texelFetch(preclassification, preclass_id).x;
  gpucast_count_texel_fetch();

  return int(preclass);
}

#endif