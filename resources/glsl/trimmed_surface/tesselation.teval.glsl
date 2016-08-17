#extension GL_NV_gpu_shader5 : enable

///////////////////////////////////////////////////////////////////////////////
// input
/////////////////////////////////////////////////////////////////////////////// 
layout(quads, equal_spacing, ccw) in;               
                                                            
flat in uint  tcIndex[];                            
flat in vec2  tcTessCoord[];              

///////////////////////////////////////////////////////////////////////////////
// output
/////////////////////////////////////////////////////////////////////////////// 
flat out uint   teIndex;                            
flat out vec2   teTessCoord;                        
flat out vec4   teNormal;                           
flat out vec4   tePosition;       
                                                                          
///////////////////////////////////////////////////////////////////////////////
// uniforms
///////////////////////////////////////////////////////////////////////////////
#include "./resources/glsl/common/camera_uniforms.glsl"                
                                                            
uniform samplerBuffer parameter_texture;            
uniform samplerBuffer attribute_texture;            

#define GPUCAST_HULLVERTEXMAP_SSBO_BINDING 1
#define GPUCAST_ATTRIBUTE_SSBO_BINDING 2

#include "./resources/glsl/common/obb_area.glsl"   
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"            

///////////////////////////////////////////////////////////////////////////////
// functions
///////////////////////////////////////////////////////////////////////////////
#include "./resources/glsl/math/horner_surface.glsl.frag"
#include "./resources/glsl/math/horner_surface_derivatives.glsl.frag"


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()                                                            
{                                                                      
  vec4 p, du, dv;                                                      

  int surface_index   = 0;
  int surface_order_u = 0;
  int surface_order_v = 0;
  retrieve_patch_data(int(tcIndex[0]), surface_index, surface_order_u, surface_order_v);
                                                                                      
  vec2 p1 = mix(tcTessCoord[0].xy, tcTessCoord[1].xy, gl_TessCoord.x); 
  vec2 p2 = mix(tcTessCoord[3].xy, tcTessCoord[2].xy, gl_TessCoord.x); 
                                                                               
  vec2 uv;                                                             
                                                                               
  uv = clamp(mix(p1, p2, gl_TessCoord.y), 0.0, 1.0);                   
                                                                               
  evaluateSurface(parameter_texture,                                   
                  surface_index,                                  
                  surface_order_u,                                
                  surface_order_v,                                
                  uv, p, du, dv);                                      
                                                                               
  tePosition  = vec4(p.xyz, 1.0);                                                                     
  teIndex     = tcIndex[0];                                            
  teTessCoord = uv;                                                    
  teNormal    = vec4(normalize(cross(du.xyz, dv.xyz)), 0.0);           
                                                                               
  vec4 nview  = gpucast_normal_matrix * teNormal;                          
  vec4 pview  = gpucast_view_matrix * gpucast_model_matrix * tePosition;       
                                                                               
  if ( dot(normalize(nview.xyz), -normalize(pview.xyz)) < 0.0f ) {     
    teNormal = -teNormal;                                              
  }                                                                    
}     