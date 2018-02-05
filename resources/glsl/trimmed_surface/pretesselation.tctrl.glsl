#extension GL_NV_gpu_shader5 : enable

///////////////////////////////////////////////////////////////////////////////                                         
// input
///////////////////////////////////////////////////////////////////////////////        
in vec3  vertex_position[];                  
in uint  vertex_index[];                    
in vec2  vertex_tesscoord[];   
      
///////////////////////////////////////////////////////////////////////////////
// output
///////////////////////////////////////////////////////////////////////////////                                                            
layout(vertices = 4) out; 

out vec3 control_position[];
out uint control_index[];
out vec2 control_tesscoord[];
out vec3 control_final_tesselation[];

///////////////////////////////////////////////////////////////////////////////
// uniforms
///////////////////////////////////////////////////////////////////////////////                                                            
uniform samplerBuffer gpucast_control_point_buffer;  
uniform samplerBuffer gpcuast_attribute_buffer;              
uniform samplerBuffer gpucast_obb_buffer;

#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"     
#include "./resources/glsl/trimming/trimming_uniforms.glsl"                                                                       
#include "./resources/glsl/common/camera_uniforms.glsl"       

layout(std430, binding = GPUCAST_FEEDBACK_BUFFER_BINDING) buffer gpucast_feedback_buffer {
  uint gpucast_feedback[];
};

///////////////////////////////////////////////////////////////////////////////
// methods
///////////////////////////////////////////////////////////////////////////////    
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"
#include "./resources/glsl/trimmed_surface/estimate_edge_length.glsl"
#include "./resources/glsl/common/obb_area.glsl"        
#include "./resources/glsl/common/conversion.glsl"       
       
   
         
                                                                                                 
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main() 
{        
  // passthrough patch information                                                                       
  control_position[gl_InvocationID]  = vertex_position[gl_InvocationID];                                               
  control_index[gl_InvocationID]     = vertex_index[gl_InvocationID];                                                  
  control_tesscoord[gl_InvocationID] = vertex_tesscoord[gl_InvocationID];                                              
     
  // project oriented boudning box to screen and estimate area          
  int obb_index     = retrieve_obb_index(int(vertex_index[gl_InvocationID]));

  float area        = calculate_obb_area(gpucast_model_view_projection_matrix, gpucast_model_view_inverse_matrix, gpucast_obb_buffer, obb_index, false);
  float area_pixels = float(gpucast_resolution.x * gpucast_resolution.y) * area;

#if GPUCAST_WRITE_DEBUG_COUNTER
  if ( vertex_index[gl_InvocationID] < GPUCAST_MAX_FEEDBACK_BUFFER_INDICES) {
    int index = int(vertex_index[gl_InvocationID]);
    atomicMax(gpucast_feedback[index], uint(area_pixels));
  }
#endif

  // derive desired tesselation based on projected area estimate
  float total_tess_level = sqrt(area_pixels) / gpucast_tesselation_max_pixel_error;
  
#if GPUCAST_SCALE_TEXXELLATION_FACTOR_TO_TRIM_RATIO
  int trim_index    = retrieve_trim_index(int(control_index[gl_InvocationID]));

  vec4 trim_domain  = texelFetch(gpucast_kd_partition, trim_index+1);
  vec4 nurbs_domain = texelFetch(gpucast_kd_partition, trim_index+2);

  float trim_size   = (trim_domain[1] - trim_domain[0]) * (trim_domain[3] - trim_domain[2]);
  float domain_size = (nurbs_domain[1] - nurbs_domain[0]) * (nurbs_domain[3] - nurbs_domain[2]);

  total_tess_level *= clamp(trim_size / domain_size, 0.0, 1.0);
#endif

#if 1
  // limit tessellation level by geometric size
  vec4 obb_max = texelFetch(gpucast_obb_buffer, obb_index + 11 + 6);

  vec3 obb_size = abs(obb_max.xyz) / 10.0;

  // assuming 180 degree turn -> twice maximum outer surface
  float approximate_area = max(max(obb_size.x * obb_size.y, obb_size.x * obb_size.z), obb_size.z * obb_size.y);
  float tessellation_geometric_limit = (approximate_area * approximate_area) / (gpucast_max_geometric_error * gpucast_max_geometric_error);

  total_tess_level = clamp(total_tess_level, 0.0, tessellation_geometric_limit);

#endif

  // assume final tesselation performs 64 subdivisions
  float pre_tess_level = total_tess_level / GPUCAST_HARDWARE_TESSELATION_LIMIT; 

  // clamp remaining pretesselations to max
  pre_tess_level = clamp(pre_tess_level, 1.0, gpucast_max_pre_tesselation);
  float final_tess_level = total_tess_level / pre_tess_level;

#if GPUCAST_SKIP_PRETESSELATION
  pre_tess_level = 1.0;
#endif

  // in low-quality shadow mode - don't bother with much tesselation
  if ( gpucast_shadow_mode == 2 ) {
    pre_tess_level = 1.0; 
    final_tess_level = total_tess_level / 16.0;
  }

  // in high-quality shadow mode - render @ 1/4 of the desired tesselation quality
  if ( gpucast_shadow_mode == 3 ) {
    pre_tess_level = 1.0;
    final_tess_level = total_tess_level / 4.0;
  }

  gl_TessLevelInner[0] = pre_tess_level;
  gl_TessLevelOuter[1] = pre_tess_level;
  gl_TessLevelOuter[3] = pre_tess_level;
  gl_TessLevelInner[1] = pre_tess_level;
  gl_TessLevelOuter[0] = pre_tess_level;
  gl_TessLevelOuter[2] = pre_tess_level;     
  
  // compute edge length to estimate tesselation
  int surface_index   = 0;
  int surface_order_u = 0;
  int surface_order_v = 0;
  retrieve_patch_data(int(vertex_index[gl_InvocationID]), surface_index, surface_order_u, surface_order_v);

  vec4 edge_lengths = estimate_edge_lengths_in_pixel(surface_index, 
                                                     gpucast_control_point_buffer, 
                                                     surface_order_u, 
                                                     surface_order_v,
                                                     gpucast_model_view_projection_matrix,
                                                     vec2(gpucast_resolution));
  edge_lengths /= pre_tess_level;

  control_final_tesselation[gl_InvocationID] = vec3(final_tess_level,
                                                    uintBitsToFloat(uint2ToUInt(uvec2(edge_lengths.xy))),
                                                    uintBitsToFloat(uint2ToUInt(uvec2(edge_lengths.zw))));
}