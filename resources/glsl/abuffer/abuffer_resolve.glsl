#define ABUFFER_ACCESS_MODE readonly

#include "resources/glsl/abuffer/abuffer_defines.glsl"

//////////////////////////////////////////////////////////////
void abuffer_mix_frag(vec4 frag_color, 
                      inout vec4 color) {
  frag_color.rgb *= frag_color.a;
  color += mix(frag_color, vec4(0.0), color.a);
}

//////////////////////////////////////////////////////////////
vec4 abuffer_shade(uint pos, float depth) {

  uvec4 data = gpucast_fragment_data[pos];
  vec3 color = vec3(unpackUnorm2x16(data.x), unpackUnorm2x16(data.y).x);

  //vec3 normal = vec3(unpackSnorm2x16(data.y).y, unpackSnorm2x16(data.z));
  //vec3 pbr = unpackUnorm4x8(data.w).xyz;
  //uint flags = bitfieldExtract(data.w, 24, 8);

  return vec4(color, 1.0);

}

//////////////////////////////////////////////////////////////
vec3 abuffer_get_color(uint pos) {
  uvec4 data = gpucast_fragment_data[pos];
  return vec3(unpackUnorm2x16(data.x), unpackUnorm2x16(data.y).x);
}

//////////////////////////////////////////////////////////////
bool abuffer_contains_fragments (out float depth) 
{
  const ivec2 frag_pos = ivec2(gl_FragCoord.xy);
  uint current = gpucast_resolution.x * frag_pos.y + frag_pos.x;
  int frag_count = 0;

  uvec2 frag = unpackUint2x32(gpucast_fragment_list[current]);
  current = frag.x;

  if (current == 0) {
    depth = 1.0;
    return false;
  } else {
    depth = unpack_depth24(frag.y);
    return true;
  }
}

//////////////////////////////////////////////////////////////
bool abuffer_blend(in vec4 opaque_color,
                   out vec4 color, 
                   inout float emissivity, 
                   float opaque_depth) 
{
  const ivec2 frag_pos = ivec2(gl_FragCoord.xy);
  uint current = gpucast_resolution.x * frag_pos.y + frag_pos.x;
  int frag_count = 0;

  vec3  composite_color = vec3(0.0);
  float composite_opacity = 0.0;

  while (frag_count < GPUCAST_ABUFFER_MAX_FRAGMENTS) {
    
    uvec2 frag = unpackUint2x32(gpucast_fragment_list[current]);
    current = frag.x;

    if (current == 0) {
      break;
    } 
    ++frag_count;
    
    float z = unpack_depth24(frag.y);
    if (z - GPUCAST_ABUFFER_ZFIGHTING_THRESHOLD > opaque_depth) { // fix depth-fighting artifacts
      break;
    }

    float alpha = float(bitfieldExtract(frag.y, 0, 8)) / 255.0;

#if USE_DEBUG_COLORS
    vec3 shaded_color = (alpha < 0.5) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
#else
    vec3 shaded_color = abuffer_get_color(current - gpucast_abuffer_list_offset);
#endif
    
    composite_color   = alpha * (1.0 - composite_opacity) * shaded_color + composite_color;
    composite_opacity = alpha * (1.0 - composite_opacity) + composite_opacity;
  }

  color = vec4( (1.0 - composite_opacity) * opaque_color.rgb + composite_color, 1.0);

  return true;
}
