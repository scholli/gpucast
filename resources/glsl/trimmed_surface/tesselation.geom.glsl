#extension GL_NV_gpu_shader5 : enable

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
#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"
#include "./resources/glsl/trimming/trimming_uniforms.glsl"
#include "./resources/glsl/trimming/pre_classification.glsl"
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"

void compute_partial_derivatives(in vec4 a, // xy_uv
                                 in vec4 b, // xy_uv
                                 in vec4 c, // xy_uv
                                 out vec2 duv_dx,
                                 out vec2 duv_dy)
{
  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;

  duv_dy = ( b.zw / ab.x - c.zw / ac.x - a.zw / ab.x + a.zw / ac.x ) / 
          ( ab.y / ab.x - ac.y / ac.x );
  duv_dx = ( b.zw - a.zw - ab.y * duv_dy ) / ab.x;
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()
{
  vec4 nurbs_domain = retrieve_patch_domain(int(teIndex[0]));
  vec2 domain_size  = vec2(nurbs_domain.z - nurbs_domain.x, nurbs_domain.w - nurbs_domain.y);

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

  
#if GPUCAST_TEXTURE_BASED_GEOMETRY_DISCARD
  ///////////////////////////////////////////////////////////
  // pre-classification discard
  ////////////////////////////////////////////////////////////
  bool discard_triangle = true;
  for ( int i = vertex_id_first; i != vertex_id_last; i += increment )
  {
#if GPUCAST_MAP_BEZIERCOORDS_TO_TESSELATION
    geometry_texcoords        = teTessCoord[i];
#else
    geometry_texcoords        =  teTessCoord[i] * domain_size + nurbs_domain.xy;
#endif
    int trim_index = retrieve_trim_index(int(teIndex[0]));
    vec4 baseinfo  = texelFetch(gpucast_kd_partition, trim_index);
    int classification_base_id = int(floatBitsToUint(baseinfo.y));
    vec4 domainbounds = texelFetch(gpucast_kd_partition, trim_index + 2);

    if (classification_base_id != 0)
    {
      int preclasstex_width  = int(floatBitsToUint(baseinfo.z));
      int preclasstex_height = int(floatBitsToUint(baseinfo.w));    

      int pre_class = pre_classify(gpucast_preclassification,
                                    classification_base_id,
                                    geometry_texcoords,
                                    domainbounds,
                                    preclasstex_width, 
                                    preclasstex_height);
      discard_triangle = discard_triangle && ( pre_class == 2);
    }
  }

  if (discard_triangle) {
    return;
  }
  ///////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
#endif

  for ( int i = vertex_id_first; i != vertex_id_last; i += increment )
  {
    gIndex      = teIndex[i];

    // write built-in input for material
    ///////////////////////////////////////////////////////
    geometry_world_position   = (gpucast_model_matrix * tePosition[i]).xyz;
    geometry_normal           = float(increment) * teNormal[i].xyz; // invert normal if necessary
#if GPUCAST_MAP_BEZIERCOORDS_TO_TESSELATION
    geometry_texcoords        = teTessCoord[i];
#else
    geometry_texcoords        = teTessCoord[i] * domain_size + nurbs_domain.xy;
#endif
    ///////////////////////////////////////////////////////
                      
    gl_Position = gpucast_model_view_projection_matrix * vec4(tePosition[i].xyz, 1.0);

    EmitVertex();                                                                               
  }                                                                                               
  EndPrimitive();  
  
#if GPUCAST_WRITE_DEBUG_COUNTER
  atomicCounterIncrement(triangle_counter);
#endif
}       