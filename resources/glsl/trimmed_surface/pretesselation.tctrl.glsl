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
out float control_final_tesselation[];

///////////////////////////////////////////////////////////////////////////////
// uniforms
///////////////////////////////////////////////////////////////////////////////                                                            
uniform samplerBuffer gpucast_parametric_buffer;  
uniform samplerBuffer gpcuast_attribute_buffer;              
uniform samplerBuffer gpucast_obb_buffer;
                                                                    
#define GPUCAST_HULLVERTEXMAP_SSBO_BINDING 1
#define GPUCAST_ATTRIBUTE_SSBO_BINDING 2

#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"                          
#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"        
#include "./resources/glsl/common/obb_area.glsl"
#include "./resources/glsl/common/camera_uniforms.glsl"                         

                                                                                                 
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
  float area        = calculate_obb_area(gpucast_model_view_projection_matrix, gpucast_model_view_inverse_matrix, gpucast_obb_buffer, obb_index);
  float area_pixels = float(gpucast_resolution.x * gpucast_resolution.y) * area;

  //// derive desired tesselation based on projected area estimate
  float total_tess_level = sqrt(area_pixels) / gpucast_tesselation_max_error;
  float pre_tess_level = clamp(total_tess_level, 1.0, gpucast_max_pre_tesselation);
  float final_tess_level = total_tess_level / pre_tess_level;

  // in low-quality shadow mode - don't bother with much tesselation
  if ( gpucast_shadow_mode == 1 ) {
    pre_tess_level = 1.0; 
    final_tess_level = total_tess_level / 16.0;
  }

  // in high-quality shadow mode - render @ 1/4 of the desired tesselation quality
  if ( gpucast_shadow_mode == 2 ) {
    pre_tess_level = 1.0;
    final_tess_level = total_tess_level / 4.0;
  }
  
  control_final_tesselation[gl_InvocationID] = final_tess_level;

  gl_TessLevelInner[0] = pre_tess_level;
  gl_TessLevelOuter[1] = pre_tess_level;
  gl_TessLevelOuter[3] = pre_tess_level;
  gl_TessLevelInner[1] = pre_tess_level;
  gl_TessLevelOuter[0] = pre_tess_level;
  gl_TessLevelOuter[2] = pre_tess_level;           
}