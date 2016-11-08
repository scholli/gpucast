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
#include "./resources/glsl/common/camera_uniforms.glsl"       

///////////////////////////////////////////////////////////////////////////////
// methods
///////////////////////////////////////////////////////////////////////////////    
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"                          
#include "./resources/glsl/common/obb_area.glsl"        
#include "./resources/glsl/common/conversion.glsl"       
       
       
///////////////////////////////////////////////////////////////////////////////
float estimate_control_edge (in samplerBuffer points,
                             in int first_point_index, 
                             in int second_point_index, 
                             in int stride, 
                             in int limit)
{
  int i = first_point_index;
  int j = second_point_index;

  float edge_length = 0;

  for (; j < limit; i += stride, j+= stride) {
    // get point in hyperspace coordinates (wx, wy, wz, w)
    vec4 p0 = texelFetchBuffer(points, i);
    vec4 p1 = texelFetchBuffer(points, j);

    // normalize to euclidian
    p0 /= p0.w;
    p1 /= p1.w;

    vec3 edge = p0.xyz - p1.xyz;
    edge_length += length(edge);
  }
  return edge_length;
}
          
          
///////////////////////////////////////////////////////////////////////////////
float estimate_control_edge_in_pixel(in samplerBuffer points,
                                     in int first_point_index, 
                                     in int second_point_index, 
                                     in int stride, 
                                     in int limit,
                                     in mat4 mvp,
                                     in vec2 resolution)
{
  int i = first_point_index;
  int j = second_point_index;

  float edge_length = 0;

  for (; j < limit; i += stride, j+= stride) {
    // get point in hyperspace coordinates (wx, wy, wz, w)
    vec4 p0 = texelFetchBuffer(points, i);
    vec4 p1 = texelFetchBuffer(points, j);

    // normalize to euclidian
    p0 /= p0.w;
    p1 /= p1.w;

    // transform to screen coordinates
    p0 = mvp * vec4(p0.xyz, 1.0);
    p1 = mvp * vec4(p1.xyz, 1.0);

    p0 /= p0.w;
    p1 /= p1.w;

    vec2 p0_device = (p0.xy * 0.5 + 0.5) * resolution;
    vec2 p1_device = (p1.xy * 0.5 + 0.5) * resolution;

    vec2 edge = p0_device.xy - p1_device.xy;
    edge = clamp(edge, vec2(0), resolution);
    edge_length += length(edge);
  }
  return edge_length;
}

///////////////////////////////////////////////////////////////////////////////
vec4 estimate_edge_lengths_in_pixel(in int base_id, 
                                    in samplerBuffer points, 
                                    in int order_u, 
                                    in int order_v,
                                    in mat4 mvp,
                                    in vec2 resolution) 
{
  float edge_length_umin = estimate_control_edge_in_pixel(points, 
                                                 base_id, 
                                                 base_id + order_u, 
                                                 order_u, 
                                                 base_id + order_u * order_v, 
                                                 mvp, 
                                                 resolution);
  float edge_length_umax = estimate_control_edge_in_pixel(points, 
                                                 base_id + (order_u - 1), 
                                                 base_id + (2 * order_u - 1), 
                                                 order_u, 
                                                 base_id + order_u * order_v, 
                                                 mvp, 
                                                 resolution);

  float edge_length_vmin = estimate_control_edge_in_pixel(points, 
                                                 base_id + order_u * (order_v-1), 
                                                 base_id + order_u * (order_v-1) + 1, 
                                                 1, 
                                                 base_id + order_u * order_v, 
                                                 mvp, 
                                                 resolution);

  float edge_length_vmax = estimate_control_edge_in_pixel(points, 
                                                 base_id, 
                                                 base_id + 1, 
                                                 1, 
                                                 base_id + order_u, 
                                                 mvp, 
                                                 resolution);

  return vec4(edge_length_umin, edge_length_umax, edge_length_vmin, edge_length_vmax);

}          
          
          
                                                                                                 
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

  //// derive desired tesselation based on projected area estimate
  float total_tess_level = sqrt(area_pixels) / gpucast_tesselation_max_error;
  float pre_tess_level = clamp(total_tess_level, 0.0, gpucast_max_pre_tesselation);
  float final_tess_level = total_tess_level / pre_tess_level;

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