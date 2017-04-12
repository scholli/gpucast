// HINT: define ABUFFER_ACCESS_MODE before including this file to change memory access write/readonly
#ifndef ABUFFER_ACCESS_MODE 
#define ABUFFER_ACCESS_MODE
#endif

#include "resources/glsl/common/camera_uniforms.glsl"

/////////////////////////////////////////////////////////////////////
// constants
/////////////////////////////////////////////////////////////////////
#define GPUCAST_ABUFFER_MAX_FRAGMENTS 300
#define GPUCAST_BLENDING_TERMINATION_THRESHOLD 0.99
#define GPUCAST_ABUFFER_ZFIGHTING_THRESHOLD 0.000001

// helper macros
#define UINT24_MAX           0xFFFFFF
#define UINT_MAX             0xFFFFFFFF

/////////////////////////////////////////////////////////////////////
// typedefs and macros
/////////////////////////////////////////////////////////////////////
// If max for uint64_t is not available
#if 1
  #define MAX64(x, y) (((x)>(y))?(x):(y))
  #define MIN64(x, y) (((x)<(y))?(x):(y))
#else
  // this is not available with old drivers
  #define MAX64(x, y) max(uint64_t(x), uint64_t(y))
  #define MIN64(x, y) min(uint64_t(x), uint64_t(y))
#endif

#define LSB64(a)             (uint32_t(a))

/////////////////////////////////////////////////////////////////////
// uniforms
/////////////////////////////////////////////////////////////////////
#define GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING         @GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING_INPUT@
#define GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING  @GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING_INPUT@
#define GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING  @GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING_INPUT@

layout (binding = GPUCAST_ABUFFER_ATOMIC_BUFFER_BINDING, offset = 0) uniform atomic_uint gpucast_abuffer_fragment_counter;

layout (std430, binding = GPUCAST_ABUFFER_FRAGMENT_LIST_BUFFER_BINDING) ABUFFER_ACCESS_MODE coherent buffer gpucast_abuffer_list {
  uint64_t gpucast_fragment_list[];
};

layout (std430, binding = GPUCAST_ABUFFER_FRAGMENT_DATA_BUFFER_BINDING) ABUFFER_ACCESS_MODE coherent buffer gpucast_abuffer_data {
  uvec4 gpucast_fragment_data[];
};

const unsigned int gpucast_abuffer_list_offset = gpucast_resolution.x * gpucast_resolution.y;

/////////////////////////////////////////////////////////////////////
// helper functions
/////////////////////////////////////////////////////////////////////
unsigned int pack_depth24(float z) {
  return (UINT_MAX - unsigned int(round(z * float(UINT24_MAX)))) << 8;
}

float unpack_depth24(unsigned int z) {
  return float((UINT_MAX - z) >> 8) / float(UINT24_MAX);
}
