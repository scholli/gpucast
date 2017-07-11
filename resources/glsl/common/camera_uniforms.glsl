#ifndef GPUCAST_GLSL_CAMERA_UNIFORMS
#define GPUCAST_GLSL_CAMERA_UNIFORMS

layout (std140) uniform gpucast_matrix_uniforms
{
  mat4 gpucast_model_matrix; // used
  mat4 gpucast_model_inverse_matrix;
  
  mat4 gpucast_model_view_matrix; // used
  mat4 gpucast_model_view_inverse_matrix;
  
  mat4 gpucast_model_view_projection_matrix; // used
  mat4 gpucast_model_view_projection_inverse_matrix;
  
  mat4 gpucast_normal_matrix;  // used
  
  mat4 gpucast_view_matrix;
  mat4 gpucast_view_inverse_matrix; // used
  
  mat4 gpucast_projection_matrix;
  mat4 gpucast_projection_inverse_matrix;
  
  mat4 gpucast_projection_view_matrix;
  mat4 gpucast_projection_view_inverse_matrix;
  
  ivec2 gpucast_resolution;

  float gpucast_clip_near;
  float gpucast_clip_far;
};

#endif