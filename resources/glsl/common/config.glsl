#ifndef GPUCAST_CAST_CONFIG
#define GPUCAST_CAST_CONFIG

#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"      

#if GPUCAST_COUNT_TEXEL_FETCHES

  int gpucast_texel_fetches = 0;

  void gpucast_count_texel_fetch() { 
    ++gpucast_texel_fetches; 
  }

#else

  void gpucast_count_texel_fetch() 
  {}

#endif

#endif // GPUCAST_CAST_CONFIG



