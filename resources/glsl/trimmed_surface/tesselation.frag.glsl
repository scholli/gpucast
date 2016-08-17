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
uniform samplerBuffer parameter_texture;      
uniform samplerBuffer attribute_texture;

uniform samplerBuffer trim_partition;
uniform samplerBuffer trim_contourlist;
uniform samplerBuffer trim_curvelist;
uniform samplerBuffer trim_curvedata;
uniform samplerBuffer trim_pointdata;
uniform samplerBuffer trim_preclassification;

uniform mat4 gua_view_inverse_matrix;

#include "./resources/glsl/common/camera_uniforms.glsl"

#define GPUCAST_HULLVERTEXMAP_SSBO_BINDING 1
#define GPUCAST_ATTRIBUTE_SSBO_BINDING 2

#include "./resources/glsl/common/obb_area.glsl"   
#include "./resources/glsl/trimmed_surface/ssbo_per_patch_data.glsl"

///////////////////////////////////////////////////////////////////////////////
// methods
///////////////////////////////////////////////////////////////////////////////    
#include "./resources/glsl/math/horner_curve.glsl"
#include "./resources/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./resources/glsl/trimming/binary_search.glsl"
#include "./resources/glsl/trimming/bisect_curve.glsl"
#include "./resources/glsl/trimming/trimming_contour_double_binary.glsl"
#include "./resources/glsl/trimming/trimming_contour_kd.glsl"


// normal is assumed to be normalized already
vec3 force_front_facing_normal(vec3 normal) 
{
  vec4 C = gpucast_view_inverse_matrix * vec4(0.0, 0.0, 0.0, 1.0);
  vec3 V = normalize(C.xyz - geometry_world_position);
  vec3 N = normal.xyz;

  if (dot(V, N) < 0.0) {
    return normal *= -1.0;
  } 
  return normal;
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void main()
{
  /////////////////////////////////////////////////////
  // 1. perform trimming based on uv value
  /////////////////////////////////////////////////////
  
  // retrieve patch information from ssbo
  vec4 nurbs_domain = retrieve_patch_domain(int(gIndex));
  int trim_index    = retrieve_trim_index(int(gIndex));

  // transform bezier coordinates to knot span of according NURBS element
  vec2 domain_size  = vec2(nurbs_domain.z - nurbs_domain.x, nurbs_domain.w - nurbs_domain.y);
  vec2 uv_nurbs     = geometry_texcoords.xy * domain_size + nurbs_domain.xy;

  // classify trimming by point-in-curve test
  int tmp = 0;
  bool trimmed      = trimming_contour_kd (trim_partition,
                                           trim_contourlist,
                                           trim_curvelist,
                                           trim_curvedata,
                                           trim_pointdata,
                                           trim_preclassification,
                                           uv_nurbs,
                                           int(trim_index), 1, tmp, 0.0001f, 16);

  // fully discard trimmed fragments
  if ( trimmed ) {
      discard;
  }

  vec3 corrected_normal = force_front_facing_normal(geometry_normal);

  out_color = vec4(corrected_normal, 1.0);

  //gl_FragDepth = gl_FragCoord.z;
}