///////////////////////////////////////////////////////////////////////////////
// input
///////////////////////////////////////////////////////////////////////////////
layout (location = 0) in vec3  in_position;   
layout (location = 1) in uint  in_index;      
layout (location = 2) in vec4  in_tesscoord;  

///////////////////////////////////////////////////////////////////////////////                                         
// output
///////////////////////////////////////////////////////////////////////////////                      
#if 1
out vec3  vertex_position;                  
out uint  vertex_index;                    
out vec2  vertex_tesscoord;             
#else
layout (xfb_offset=0)  out vec3  transform_position;
layout (xfb_offset=12) out uint  transform_index;
layout (xfb_offset=16) out vec2  transform_tesscoord;
layout (xfb_offset=24) out float transform_final_tesselation;
#endif
///////////////////////////////////////////////////////////////////////////////                                         
// uniforms
///////////////////////////////////////////////////////////////////////////////   



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////                                                   
void main()                                
{              
#if 1                       
  vertex_position  = in_position;                   
  vertex_index     = in_index;                      
  vertex_tesscoord = in_tesscoord.xy;               
#else
  transform_position  = in_position;                   
  transform_index     = in_index;                      
  transform_tesscoord = in_tesscoord.xy;       
  transform_final_tesselation = 3.0;        
#endif
} 