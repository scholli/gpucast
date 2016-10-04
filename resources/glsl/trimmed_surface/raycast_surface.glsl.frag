#extension GL_EXT_gpu_shader4 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_shader_storage_buffer_object : enable
#extension GL_ARB_separate_shader_objects : enable 

/****************************************************
* choose type of root finder for trimming
****************************************************/
precision highp float;

/*******************************************************************************
 *   trim_index : start index in trim acceleration for current surface
 *   data_index : start index in control point buffer for current surface
 *   uvrange    : parameter range of surface [umin, umax, vmin, vmax]
 *   trimtype   : 0 = inner is trimmed, 1 outer is trimmed
 ******************************************************************************/

/*******************************************************************************
 * cubemap for reflections
 ******************************************************************************/
uniform sampler2D   spheremap;
uniform sampler2D   diffusemap;

uniform int diffusemapping;
uniform int spheremapping;

uniform float shininess;
uniform float opacity;
uniform vec3 mat_ambient;
uniform vec3 mat_diffuse;
uniform vec3 mat_specular;

/*******************************************************************************
 * VARYINGS :
 *   v_modelcoord : rasterized model coordinates
 ******************************************************************************/
in vec4 v_modelcoord;
in vec4 frag_texcoord;

flat in int trim_index;
flat in int trim_approach;
flat in int trim_type;

flat in int obb_index;
flat in int data_index;
flat in int order_u;
flat in int order_v;

flat in vec4 uvrange;

/*******************************************************************************
 * UNIFORMS :
 ******************************************************************************/
#include "./resources/glsl/common/camera_uniforms.glsl"
#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"
#include "./resources/glsl/trimmed_surface/raycasting_uniforms.glsl"
#include "./resources/glsl/trimming/trimming_uniforms.glsl"

/*******************************************************************************
 * OUTPUT
 ******************************************************************************/
layout (location = 0) out vec4  out_color;
layout (depth_any)    out float gl_FragDepth;

/*******************************************************************************
 * include functions
 ******************************************************************************/
#include "./resources/glsl/base/compute_depth.frag"
#include "./resources/glsl/math/adjoint.glsl.frag"
#include "./resources/glsl/math/euclidian_space.glsl.frag"
#include "./resources/glsl/math/horner_surface.glsl.frag"
#include "./resources/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./resources/glsl/math/newton_surface.glsl.frag"
#include "./resources/glsl/math/raygeneration.glsl.frag" 
#include "./resources/glsl/trimming/trimming_contour_double_binary.glsl"
#include "./resources/glsl/trimming/trimming_double_binary.glsl"
#include "./resources/glsl/trimming/trimming_contour_kd.glsl"
#include "./resources/glsl/trimming/trimming_loop_lists.glsl"
#include "./resources/glsl/trimmed_surface/shade_phong_fresnel.glsl.frag"


/*******************************************************************************
 * raycast shader for trimmed rational bezier surfaces
 ******************************************************************************/
void main(void)
{
  /*********************************************************************
  * Ray generation
  *********************************************************************/
  vec3 n1, n2;
  float d1, d2;
  raygen(v_modelcoord, gpucast_model_view_inverse_matrix, n1, n2, d1, d2);

  /*********************************************************************
  * Surface intersection
  *********************************************************************/
  vec2 uv = vec2(frag_texcoord[0], frag_texcoord[1]);
  
  vec4 p  = v_modelcoord;
  vec4 du = vec4(0.0);
  vec4 dv = vec4(0.0);

  if ( bool(gpucast_enable_newton_iteration) ) 
  {
    bool surface_hit = newton(uv, 
                              gpucast_raycasting_error_tolerance, 
                              gpucast_raycasting_iterations, 
                              gpucast_control_point_buffer, 
                              data_index, 
                              order_u, 
                              order_v, 
                              n1, n2, d1, d2, p, du, dv);
    if ( !surface_hit ) 
    {
      discard;
    }
  }
  
  vec3 normal = normalize(cross(normalize(du.xyz), normalize(dv.xyz)));
  
  /*********************************************************************
   * Trimming process
   *********************************************************************/
  
  // transform in NURBS parameter coordinates
  uv[0] = uvrange[0] + uv[0] * (uvrange[1] - uvrange[0]);
  uv[1] = uvrange[2] + uv[1] * (uvrange[3] - uvrange[2]);
  int trim_iterations = 0;
  vec4 debug = vec4(0.0);

  bool is_trimmed = false;

  if (trim_approach == 0)
  {
    is_trimmed = false;
  }

  if (trim_approach == 1)
  {
    is_trimmed = trimming_double_binary ( gpucast_bp_trimdata, 
                                          gpucast_bp_celldata, 
                                          gpucast_bp_curvelist, 
                                          gpucast_bp_curvedata, 
                                          gpucast_preclassification,
                                          uv, 
                                          trim_index, 
                                          trim_type, 
                                          trim_iterations, 
                                          gpucast_trim_error_tolerance, 
                                          gpucast_trimming_max_bisections );
  }
  
  if (trim_approach == 2) {
    is_trimmed= trimming_contour_double_binary ( gpucast_cmb_partition, 
                                                 gpucast_cmb_contourlist,
                                                 gpucast_cmb_curvelist,
                                                 gpucast_cmb_curvedata,
                                                 gpucast_cmb_pointdata,
                                                 gpucast_preclassification,
                                                 uv, 
                                                 trim_index,
                                                 trim_type, 
                                                 trim_iterations, 
                                                 gpucast_trim_error_tolerance, 
                                                 gpucast_trimming_max_bisections );
  }

  if (trim_approach == 3) {
    is_trimmed = trimming_contour_kd(gpucast_kd_partition,
                                     gpucast_kd_contourlist,
                                     gpucast_kd_curvelist,
                                     gpucast_kd_curvedata,
                                     gpucast_kd_pointdata,
                                     gpucast_preclassification,
                                     uv,
                                     trim_index,
                                     trim_type,
                                     trim_iterations,
                                     gpucast_trim_error_tolerance, 
                                     gpucast_trimming_max_bisections);
  }

  if (trim_approach == 4) {
    is_trimmed = trimming_loop_list(uv, trim_index, gpucast_preclassification);
  }

  if ( is_trimmed )
  {
    discard;
  }

  /*********************************************************************
   * depth correction
   *********************************************************************/
  vec4 p_world = gpucast_model_view_matrix * vec4(p.xyz, 1.0);
  float corrected_depth = compute_depth ( p_world, gpucast_clip_near, gpucast_clip_far );
  gl_FragDepth = corrected_depth;
  
  /*********************************************************************
   * Shading process
   ********************************************************************/
  if (bool(gpucast_enable_newton_iteration)) 
  {
    out_color = shade_phong_fresnel(p_world, 
                                    normalize((gpucast_normal_matrix * vec4(normal, 0.0)).xyz), 
                                    vec4(1.0, 1.0, 1.0, 1.0),
                                    //vec3(0.1), vec3(0.8, 0.5, 0.2), vec3(1.0),
                                    mat_ambient, mat_diffuse, mat_specular,
                                    shininess,
                                    opacity,
                                    bool(spheremapping),
                                    spheremap,
                                    bool(diffusemapping),
                                    diffusemap);
    out_color = vec4(float(trim_index)/800);
    out_color = vec4(uv/100, 1.0, 1.0);
  } else {
    out_color = vec4(frag_texcoord.xy, 0.0, 1.0);
  }
}





