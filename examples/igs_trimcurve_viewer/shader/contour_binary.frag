#extension GL_ARB_separate_shader_objects : enable 
#extension GL_NV_gpu_shader5 : enable
     
in vec2     uv_coord; 
     
uniform int trimid;

uniform samplerBuffer sampler_partition;
uniform samplerBuffer sampler_contourlist;
uniform samplerBuffer sampler_curvelist;
uniform samplerBuffer sampler_curvedata;
uniform samplerBuffer sampler_pointdata;

uniform sampler1D     transfertexture;

layout (location = 0) out vec4 outcolor; 


#include "resources/glsl/trimming/trimming_contour_double_binary.glsl"


void main(void) 
{ 
  int iterations = 0;
  vec4 debug = vec4 ( 0.0 );
  bool trimmed = trimming_contour_double_binary(sampler_partition,
                                                sampler_contourlist,
                                                sampler_curvelist,
                                                sampler_curvedata,
                                                sampler_pointdata,
                                                uv_coord, 
                                                trimid, 
                                                1, 
                                                iterations, 
                                                0.00001, 
                                                16 );

  if ( trimmed ) 
  {
    outcolor = vec4(1.0, 0.0, 0.0, 1.0 );
  } else {
    outcolor = vec4(0.0, 1.0, 0.0, 1.0 );
  }
}



