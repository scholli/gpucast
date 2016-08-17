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

// todo:include "resources/shaders/common/gua_global_variable_declaration.glsl"

///////////////////////////////////////////////////////////////////////////////
// guacamole vertex output interface
///////////////////////////////////////////////////////////////////////////////
out vec3 geometry_world_position;
out vec3 geometry_normal;
out vec2 geometry_texcoords;

///////////////////////////////////////////////////////////////////////////////
// built-in output 
///////////////////////////////////////////////////////////////////////////////     
// todo:include "resources/shaders/common/gua_vertex_shader_output.glsl"
                                                                          
///////////////////////////////////////////////////////////////////////////////
// uniforms
///////////////////////////////////////////////////////////////////////////////                                                                      
#include "./resources/glsl/common/camera_uniforms.glsl"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()
{
  // force vertex order to be face forward! 
  mat4 modelview = gpucast_view_matrix * gpucast_model_matrix;

  vec3 a_view_space = (modelview * tePosition[0]).xyz;
  vec3 b_view_space = (modelview * tePosition[1]).xyz;
  vec3 c_view_space = (modelview * tePosition[2]).xyz;

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

  for ( int i = vertex_id_first; i != vertex_id_last; i = i + increment )
  {
    gIndex      = teIndex[i];

    // write built-in input for material
    ///////////////////////////////////////////////////////

    vec4 world_normal    = gpucast_normal_matrix * vec4 (teNormal[i].xyz, 0.0);

    geometry_world_position   = (gpucast_model_matrix * tePosition[i]).xyz;
    geometry_normal           = normalize ( gpucast_normal_matrix * vec4 (teNormal[i].xyz, 0.0) ).xyz;
    geometry_texcoords        = teTessCoord[i];
    ///////////////////////////////////////////////////////
                      
    gl_Position = gpucast_projection_matrix * gpucast_view_matrix * vec4(geometry_world_position.xyz, 1.0);

    EmitVertex();                                                                               
  }                                                                                               
  EndPrimitive();                                                                                         
}       