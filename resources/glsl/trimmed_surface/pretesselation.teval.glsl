#extension GL_NV_gpu_shader5 : enable

///////////////////////////////////////////////////////////////////////////////                                         
// input
///////////////////////////////////////////////////////////////////////////////
layout(quads, equal_spacing, ccw) in;               
                                                            
in vec3  control_position[];                         
in uint  control_index[];                            
in vec2  control_tesscoord[];                        
in vec3 control_final_tesselation[];
        
///////////////////////////////////////////////////////////////////////////////
// output
///////////////////////////////////////////////////////////////////////////////                                                            
out vec3 eval_position;                           
out uint eval_index;                              
out vec2 eval_tesscoord;     
out vec3 eval_final_tesselation;                     
         
///////////////////////////////////////////////////////////////////////////////                                                   
// uniforms                                         
///////////////////////////////////////////////////////////////////////////////                                                      
uniform samplerBuffer parameter_texture;    
uniform samplerBuffer attribute_texture;              

#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl" 
#include "./resources/glsl/common/camera_uniforms.glsl"

///////////////////////////////////////////////////////////////////////////////
// functions
///////////////////////////////////////////////////////////////////////////////
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"         
#include "./resources/glsl/math/horner_surface.glsl.frag"  
#include "./resources/glsl/common/conversion.glsl"



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()                                                            
{                                                                     
  vec2 p1 = mix(control_tesscoord[0].xy, control_tesscoord[1].xy, gl_TessCoord.x); 
  vec2 p2 = mix(control_tesscoord[3].xy, control_tesscoord[2].xy, gl_TessCoord.x); 
  vec2 uv = clamp(mix(p1, p2, gl_TessCoord.y), 0.0, 1.0);                                                                                 

  int surface_index   = 0;
  int surface_order_u = 0;
  int surface_order_v = 0;
  retrieve_patch_data(int(control_index[0]), surface_index, surface_order_u, surface_order_v);

  vec4 puv, du, dv;                                                    
  evaluateSurface(parameter_texture,                                   
                  surface_index,                                  
                  surface_order_u,                                
                  surface_order_v,                                
                  uv,                                                  
                  puv);                                                
                                                                               
  eval_position  = puv.xyz;                                               
  eval_index     = control_index[0];                                            
  eval_tesscoord = uv;                                       
  eval_final_tesselation = control_final_tesselation[0];
                    
}  


