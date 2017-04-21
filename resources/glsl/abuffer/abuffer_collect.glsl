#include "resources/glsl/abuffer/abuffer_defines.glsl"

///////////////////////////////////////////////////////////////////////////////
bool abuffer_insert(float depth,
                    float alpha,
                    float emissivity, 
                    float roughness, 
                    float metalness, 
                    vec3 color, 
                    vec3 normal, 
                    bool passthrough)
{
  const ivec2 frag_pos = ivec2(gl_FragCoord.xy);

  uint ctr = atomicCounterIncrement(gpucast_abuffer_fragment_counter);
  float frag_alpha = clamp(alpha, 0.0, 1.0);

  uint record_location = ctr + gpucast_abuffer_list_offset;
  gpucast_fragment_list[record_location] = 0ul;
  memoryBarrier();

  // pack depth and alpha
  uint z_ordered = bitfieldInsert(pack_depth24(depth),
                                  uint(round(frag_alpha * 255.0)), 0, 8);

  uint64_t old, record = packUint2x32(uvec2(record_location, z_ordered));

  uint pos = gpucast_resolution.x * frag_pos.y + frag_pos.x; // start of the search
  int frag_ctr = 0;
  float accum_alpha = 0;
  bool success = false;

  uint64_t assumed;

  // insert record to the linked-list
  while (true) {
#ifdef GL_NV_shader_atomic_int64
    old = atomicMax(gpucast_fragment_list[pos], record);
#else
    old = gpucast_fragment_list[pos];
    do {
      assumed = old;
      old = atomicCompSwap(gpucast_fragment_list[pos], assumed, MAX64(record, assumed));
    } while (assumed != old);
#endif
    if (old == 0) {
      success = true;
      break;
    } else {
      if (old > record) { // go to next
        pos = LSB64(old);

        // early termination
        if (!success) {
          float current_frag_alpha = float(bitfieldExtract(unpackUint2x32(old).y, 0, 8)) / 255.0;
          accum_alpha += mix(current_frag_alpha, 0.0, accum_alpha);
          if (accum_alpha > GPUCAST_BLENDING_TERMINATION_THRESHOLD) {
            break;
          }
        }
      }
      else { // inserted
        pos = LSB64(record);
        record = old;
        success = true;
      }
    }
    // stop if maximal number of fragments is reached
    if (frag_ctr++ >= GPUCAST_ABUFFER_MAX_FRAGMENTS) {
      break; 
    }
  }

  if (success) {
    // write data
    uint pbr = packUnorm4x8(vec4(emissivity, roughness, metalness, 0.0));
    pbr = bitfieldInsert(pbr, ((passthrough)?1u:0u), 24, 8);

    uint col_norm = bitfieldInsert(packUnorm2x16(color.bb),
                                   packSnorm2x16(normal.xx), 16, 16);

    gpucast_fragment_data[ctr] = uvec4(packUnorm2x16(color.rg), col_norm,
                                       packSnorm2x16(normal.yz), pbr);
  }
  return success;
}

///////////////////////////////////////////////////////////////////////////////
void submit_fragment(float depth,
                     float alpha,
                     sampler2D gbuffer_depth, 
                     float emissivity, 
                     float roughness, 
                     float metalness, 
                     vec3 color,  
                     vec3 normal, 
                     bool passthrough)
{
  // retrieve gbuffer depth
  float z = texelFetch(gbuffer_depth, ivec2(gl_FragCoord.xy), 0).x;

  // transparent fragment is hidden
  if (depth > z) {
    //discard; //disable this optimization
  }

  // transparent fragment is barely visible -> discard
  if (alpha < 1.0 - GPUCAST_BLENDING_TERMINATION_THRESHOLD) {
    discard;
  }

  // transparent fragment is nearly opaque -> pass to gbuffer
  if (alpha > GPUCAST_BLENDING_TERMINATION_THRESHOLD) {
    // normal write 
  }
  else {
    // insert transparent fragment into abuffer
    if (abuffer_insert(depth, alpha, emissivity, roughness, metalness, color, normal, passthrough)) {
      discard;
    }
  }
}

