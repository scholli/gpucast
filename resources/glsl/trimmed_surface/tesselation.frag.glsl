#extension GL_NV_gpu_shader5 : enable

///////////////////////////////////////////////////////////////////////////////
// constants
///////////////////////////////////////////////////////////////////////////////    
precision highp float;

///////////////////////////////////////////////////////////////////////////////
// input
///////////////////////////////////////////////////////////////////////////////    
flat in uint gIndex;

in vec3 geometry_world_position;
in vec3 geometry_normal;
in vec2 geometry_texcoords;

///////////////////////////////////////////////////////////////////////////////
// output
///////////////////////////////////////////////////////////////////////////////    
layout (location = 0) out vec4  out_color;
//layout (depth_any)    out float gl_FragDepth;

///////////////////////////////////////////////////////////////////////////////
// uniforms
///////////////////////////////////////////////////////////////////////////////  
uniform samplerBuffer gpucast_parametric_buffer;   
uniform samplerBuffer gpcuast_attribute_buffer;      
uniform samplerBuffer gpucast_obb_buffer;       

#include "./resources/glsl/common/camera_uniforms.glsl"
#include "./resources/glsl/trimming/trimming_uniforms.glsl"
#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"

///////////////////////////////////////////////////////////////////////////////
// shading and material
///////////////////////////////////////////////////////////////////////////////
uniform sampler2D   spheremap;
uniform sampler2D   diffusemap;

uniform int diffusemapping;
uniform int spheremapping;

uniform float shininess;
uniform float opacity;
uniform vec3 mat_ambient;
uniform vec3 mat_diffuse;
uniform vec3 mat_specular;

#include "./resources/glsl/trimmed_surface/shade_phong_fresnel.glsl.frag"

///////////////////////////////////////////////////////////////////////////////
// methods
///////////////////////////////////////////////////////////////////////////////    
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"
#include "./resources/glsl/common/obb_area.glsl"   
#include "./resources/glsl/math/horner_curve.glsl"
#include "./resources/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./resources/glsl/trimming/binary_search.glsl"
#include "./resources/glsl/trimming/bisect_curve.glsl" 
#include "./resources/glsl/trimming/trimming_contour_double_binary.glsl"
#include "./resources/glsl/trimming/trimming_contour_kd.glsl"
#include "./resources/glsl/trimming/trimming_double_binary.glsl"
#include "./resources/glsl/trimming/trimming_loop_lists.glsl"


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()
{
  // retrieve NURBS patch information
  vec4 nurbs_domain = retrieve_patch_domain(int(gIndex));
  int trim_index    = retrieve_trim_index(int(gIndex));

  // transform Bezier uv-coordinates to nurbs knot span
  vec2 domain_size  = vec2(nurbs_domain.z - nurbs_domain.x, nurbs_domain.w - nurbs_domain.y);
  vec2 uv_nurbs     = geometry_texcoords.xy * domain_size + nurbs_domain.xy;
  
  bool is_trimmed = false;
  int trim_type = retrieve_trim_type(int(gIndex));
  int tmp = 0;

  if (gpucast_trimming_method == 0)
  {
    is_trimmed = false;
  }

  if (gpucast_trimming_method == 1)
  {
    is_trimmed = trimming_double_binary ( gpucast_bp_trimdata, 
                                          gpucast_bp_celldata, 
                                          gpucast_bp_curvelist, 
                                          gpucast_bp_curvedata, 
                                          gpucast_preclassification,
                                          uv_nurbs, 
                                          trim_index, 
                                          trim_type, 
                                          tmp,
                                          gpucast_trimming_error_tolerance, 
                                          gpucast_trimming_max_bisections );
  }
  
  if (gpucast_trimming_method == 2) {
    is_trimmed= trimming_contour_double_binary ( gpucast_cmb_partition, 
                                                 gpucast_cmb_contourlist,
                                                 gpucast_cmb_curvelist,
                                                 gpucast_cmb_curvedata,
                                                 gpucast_cmb_pointdata,
                                                 gpucast_preclassification,
                                                 uv_nurbs, 
                                                 trim_index,
                                                 trim_type, 
                                                 tmp,
                                                 gpucast_trimming_error_tolerance, 
                                                 gpucast_trimming_max_bisections );
  }

  if (gpucast_trimming_method == 3) {
    is_trimmed = trimming_contour_kd(gpucast_kd_partition,
                                     gpucast_kd_contourlist,
                                     gpucast_kd_curvelist,
                                     gpucast_kd_curvedata,
                                     gpucast_kd_pointdata,
                                     gpucast_preclassification,
                                     uv_nurbs,
                                     trim_index,
                                     trim_type,
                                     tmp,
                                     gpucast_trimming_error_tolerance, 
                                     gpucast_trimming_max_bisections);
  }

  if (gpucast_trimming_method == 4) {
    is_trimmed = trimming_loop_list(uv_nurbs, trim_index, gpucast_preclassification);
  }

  if ( is_trimmed )
  {
    discard;
  }

  if ( !is_trimmed && gpucast_enable_counting != 0) {
    atomicCounterIncrement(fragment_counter);
  }

  vec4 normal_world     = gpucast_normal_matrix * vec4(geometry_normal, 0.0);
  vec4 viewer           = gpucast_view_inverse_matrix * vec4(0.0, 0.0, 0.0, 1.0);

  out_color = shade_phong_fresnel(vec4(geometry_world_position, 1.0), 
                                  normalize(normal_world.xyz), 
                                  normalize(viewer.xyz),
                                  vec4(0.0, 0.0, 10000.0, 1.0),
                                  mat_ambient, mat_diffuse, mat_specular,
                                  shininess,
                                  opacity,
                                  bool(spheremapping),
                                  spheremap,
                                  bool(diffusemapping),
                                  diffusemap);
}