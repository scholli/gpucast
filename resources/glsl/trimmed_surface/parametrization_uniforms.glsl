#ifndef GPUCAST_GLSL_RAYCASTING_PARAMETRIZATION
#define GPUCAST_GLSL_RAYCASTING_PARAMETRIZATION

// tesselation
uniform float gpucast_max_pre_tesselation;
uniform float gpucast_tesselation_max_error;

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
uniform float gpucast_trim_error_tolerance;
uniform int   gpucast_trimming_max_bisections;

#endif