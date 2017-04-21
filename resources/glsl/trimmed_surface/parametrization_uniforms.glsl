#ifndef GPUCAST_GLSL_RAYCASTING_PARAMETRIZATION
#define GPUCAST_GLSL_RAYCASTING_PARAMETRIZATION

#define GPUCAST_HULLVERTEXMAP_SSBO_BINDING        @GPUCAST_HULLVERTEXMAP_SSBO_BINDING_INPUT@
#define GPUCAST_ATTRIBUTE_SSBO_BINDING            @GPUCAST_ATTRIBUTE_SSBO_BINDING_INPUT@
#define GPUCAST_ATOMIC_COUNTER_BINDING            @GPUCAST_ATOMIC_COUNTER_BINDING_INPUT@
#define GPUCAST_FEEDBACK_BUFFER_BINDING           @GPUCAST_FEEDBACK_BUFFER_BINDING_INPUT@

#define GPUCAST_HOLE_FILLING                      @GPUCAST_HOLE_FILLING_INPUT@
#define GPUCAST_HOLE_FILLING_THRESHOLD            0.001

#define GPUCAST_MAX_FEEDBACK_BUFFER_INDICES       @GPUCAST_MAX_FEEDBACK_BUFFER_INDICES_INPUT@
#define GPUCAST_SECOND_PASS_TRIANGLE_TESSELATION  @GPUCAST_SECOND_PASS_TRIANGLE_TESSELATION_INPUT@
#define GPUCAST_WRITE_DEBUG_COUNTER               @GPUCAST_WRITE_DEBUG_COUNTER_INPUT@
#define GPUCAST_ANTI_ALIASING_MODE                @GPUCAST_ANTI_ALIASING_MODE_INPUT@
#define GPUCAST_TRIMMING_COVERAGE_ESTIMATION      3

#define GPUCAST_MAP_BEZIERCOORDS_TO_TESSELATION   0
#define GPUCAST_TEXTURE_BASED_GEOMETRY_DISCARD    0

// tesselation
uniform float gpucast_max_pre_tesselation;
uniform float gpucast_tesselation_max_pixel_error;
uniform float gpucast_max_geometric_error;

// gpucast_shadow_mode = 0 // no shadow 
// gpucast_shadow_mode = 1 // coarse shadow mode 1/16th of full tesselation
// gpucast_shadow_mode = 2 // fine shadow mode 1/4th of full tesselation
uniform int   gpucast_shadow_mode; 

// ray casting
uniform int   gpucast_enable_newton_iteration;
uniform float gpucast_raycasting_error_tolerance;
uniform int   gpucast_raycasting_iterations;

// trimming 
// 0 - disabled
// 1 - classic double binary partition
// 2 - contour binary partition
// 3 - contour kd partition
// 4 - loop list partition
uniform int   gpucast_trimming_method;
uniform float gpucast_trimming_error_tolerance;
uniform int   gpucast_trimming_max_bisections;

layout(binding = GPUCAST_ATOMIC_COUNTER_BINDING, offset = 0) uniform atomic_uint  triangle_counter;
layout(binding = GPUCAST_ATOMIC_COUNTER_BINDING, offset = 4) uniform atomic_uint  fragment_counter;
layout(binding = GPUCAST_ATOMIC_COUNTER_BINDING, offset = 8) uniform atomic_uint  culled_triangles_counter;
layout(binding = GPUCAST_ATOMIC_COUNTER_BINDING, offset = 12) uniform atomic_uint trimmed_fragments_counter;

#endif