///////////////////////////////////////////////////////////////////////////////
// input
///////////////////////////////////////////////////////////////////////////////                                                               
layout(triangles) in;                              
layout(triangle_strip, max_vertices = 3) out;                    
                                                                              
flat in uint teIndex[3];                           
flat in vec2 teTessCoord[3];                       
flat in vec4 teNormal[3];                          
flat in vec4 tePosition[3];     
 
///////////////////////////////////////////////////////////////////////////////
// output
///////////////////////////////////////////////////////////////////////////////                                                               
flat out uint gIndex;

///////////////////////////////////////////////////////////////////////////////
// guacamole vertex output interface
///////////////////////////////////////////////////////////////////////////////
out vec3 geometry_world_position;
out vec3 geometry_normal;
out vec2 geometry_texcoords;
                                                                   
///////////////////////////////////////////////////////////////////////////////
// uniforms
///////////////////////////////////////////////////////////////////////////////                                                                      
#include "./resources/glsl/common/camera_uniforms.glsl"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()
{
  // force vertex order to be face forward to prevent backface culling!!!
  vec3 a_view_space = (gpucast_model_view_matrix * tePosition[0]).xyz;
  vec3 b_view_space = (gpucast_model_view_matrix * tePosition[1]).xyz;
  vec3 c_view_space = (gpucast_model_view_matrix * tePosition[2]).xyz;

  vec3 normal_view_space = cross(normalize(b_view_space - a_view_space), normalize(c_view_space - b_view_space));

  int vertex_id_first = 0;
  int vertex_id_last  = 3;
  int increment       = 1;

  bool invert_vertex_order = dot(normal_view_space, normalize(-a_view_space)) <= 0.0;

  if (invert_vertex_order) {
    vertex_id_first = 2;
    vertex_id_last  = -1;
    increment       = -1;
  }


  for ( int i = vertex_id_first; i != vertex_id_last; i += increment )
  {
    gIndex      = teIndex[i];

    // write built-in input for material
    ///////////////////////////////////////////////////////
    geometry_world_position   = (gpucast_model_matrix * tePosition[i]).xyz;
    geometry_normal           = teNormal[i].xyz;
    geometry_texcoords        = teTessCoord[i];
    ///////////////////////////////////////////////////////
                      
    gl_Position = gpucast_model_view_projection_matrix * vec4(tePosition[i].xyz, 1.0);

    EmitVertex();                                                                               
  }                                                                                               
  EndPrimitive();                                                                                         
}       