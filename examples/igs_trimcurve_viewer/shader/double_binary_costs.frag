#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable

in vec2     uv_coord; 

uniform int trimid;

uniform samplerBuffer trimdata;
uniform samplerBuffer celldata;
uniform samplerBuffer curvelist;
uniform samplerBuffer curvedata;
uniform sampler1D     transfertexture;

layout (location = 0) out vec4 outcolor; 

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/math/transfer_function.glsl"
#include "resources/glsl/trimming/trimming_double_binary.glsl"

void main(void) 
{ 
  int iterations = 0;
  bool trimmed = trimming_double_binary ( trimdata, celldata, curvelist, curvedata, uv_coord, trimid, 1, iterations, 0.00001, 16 );

  outcolor = transfer_function(clamp(float(gpucast_texel_fetches)/25.0, 0.0, 1.0));
}



