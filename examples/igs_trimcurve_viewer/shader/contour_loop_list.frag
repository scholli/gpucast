#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_shader_storage_buffer_object : enable

#include "resources/glsl/common/config.glsl"
#include "resources/glsl/math/transfer_function.glsl"
#include "resources/glsl/trimming/trimming_loop_lists.glsl"

in vec2     uv_coord; 

uniform int trim_index;
uniform int show_costs;

layout (location = 0) out vec4 outcolor; 

void main(void) 
{ 
  int iterations = 0;

  bool trimmed = trimming_loop_list(uv_coord, trim_index);

  if ( show_costs != 0 )
  {
    outcolor = transfer_function(clamp(float(gpucast_texel_fetches)/64.0, 0.0, 1.0));
  } else {
    if ( trimmed ) 
    {
      //outcolor = vec4(1.0, 0.0, 0.0, 1.0 ) * (float(gpucast_texel_fetches)/64.0 + 0.1);
      outcolor = vec4(0.0);
    } else {
      //outcolor = vec4(0.0, 1.0, 0.0, 1.0 ) * (float(gpucast_texel_fetches)/64.0 + 0.1);
      outcolor = vec4(1.0);
    }
  } 
}



