#ifndef GPUCAST_GLSL_CAMERA_UNIFORMS
#define GPUCAST_GLSL_CAMERA_UNIFORMS

// gpucast matrices
uniform mat4 gpucast_model_matrix; // used
uniform mat4 gpucast_model_inverse_matrix;

uniform mat4 gpucast_model_view_matrix; // used
uniform mat4 gpucast_model_view_inverse_matrix;

uniform mat4 gpucast_model_view_projection_matrix; // used
uniform mat4 gpucast_model_view_projection_inverse_matrix;

uniform mat4 gpucast_normal_matrix;  // used

uniform mat4 gpucast_view_matrix;
uniform mat4 gpucast_view_inverse_matrix; // used

uniform mat4 gpucast_projection_matrix;
uniform mat4 gpucast_projection_inverse_matrix;

uniform mat4 gpucast_projection_view_matrix;
uniform mat4 gpucast_projection_view_inverse_matrix;
  
uniform ivec2 gpucast_resolution;

uniform float gpucast_clip_near;
uniform float gpucast_clip_far;

#endif