#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_int64 : enable

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
uniform samplerBuffer gpucast_control_point_buffer;   
uniform samplerBuffer gpcuast_attribute_buffer;      
uniform samplerBuffer gpucast_obb_buffer;   
    
// optimization and a-buffer
uniform sampler2D gpucast_depth_buffer;     

#include "./resources/glsl/common/camera_uniforms.glsl"
#include "./resources/glsl/trimming/trimming_uniforms.glsl"
#include "./resources/glsl/trimmed_surface/parametrization_uniforms.glsl"
#include "./resources/glsl/abuffer/abuffer_collect.glsl"

///////////////////////////////////////////////////////////////////////////////
// shading and material
///////////////////////////////////////////////////////////////////////////////
uniform sampler2D   spheremap;
uniform sampler2D   diffusemap;

uniform int diffusemapping;
uniform int spheremapping;

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

  /////////////////////////////////////////////////////////////////////////////
  // no anti-aliasing or MSAA
  /////////////////////////////////////////////////////////////////////////////
#if GPUCAST_ANTI_ALIASING_MODE == 0 || GPUCAST_ANTI_ALIASING_MODE == 6

#if GPUCAST_MAP_BEZIERCOORDS_TO_TESSELATION
  vec2 uv_nurbs     = geometry_texcoords.xy * domain_size + nurbs_domain.xy;
#else
  vec2 uv_nurbs     = geometry_texcoords.xy;
#endif

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

  vec4 normal_world     = gpucast_normal_matrix * vec4(geometry_normal, 0.0);
  vec4 viewer           = gpucast_view_inverse_matrix * vec4(0.0, 0.0, 0.0, 1.0);

  out_color = shade_phong_fresnel(vec4(geometry_world_position, 1.0), 
                                  normalize(normal_world.xyz), 
                                  normalize(viewer.xyz),
                                  vec4(0.0, 0.0, 10000.0, 1.0),
                                  gpucast_material_ambient.rgb, 
                                  gpucast_material_diffuse.rgb, 
                                  gpucast_material_specular.rgb,
                                  gpucast_shininess,
                                  gpucast_opacity,
                                  bool(spheremapping),
                                  spheremap,
                                  bool(diffusemapping),
                                  diffusemap);

    submit_fragment(gl_FragCoord.z,
                    gpucast_opacity,
                    gpucast_depth_buffer, 
                    1.0, 1.0, 1.0,  // pbr
                    out_color.rgb,  
                    normal_world.rgb, 
                    false);
#endif
  
  /////////////////////////////////////////////////////////////////////////////
  // prefiltered coverage estimation
  /////////////////////////////////////////////////////////////////////////////
#if GPUCAST_ANTI_ALIASING_MODE == 1

#if GPUCAST_MAP_BEZIERCOORDS_TO_TESSELATION
  vec2 uv           = geometry_texcoords.xy;
  vec2 uv_nurbs     = uv * domain_size + nurbs_domain.xy;

  vec2 duv_dx       = dFdx(geometry_texcoords.xy);
  vec2 duv_dy       = dFdy(geometry_texcoords.xy);
  
  duv_dx           *= domain_size;
  duv_dy           *= domain_size;
#else
  vec2 uv_nurbs     = geometry_texcoords.xy;
  vec2 duv_dx       = dFdx(geometry_texcoords.xy);
  vec2 duv_dy       = dFdy(geometry_texcoords.xy);
#endif

  bool is_trimmed = false;
  int trim_type = retrieve_trim_type(int(gIndex));
  int tmp = 0;

  float coverage = 1.0;

  if (gpucast_trimming_method == 0)
  {
    coverage = 1.0;
  }

  if (gpucast_trimming_method == 1)
  {
    coverage = trimming_double_binary_coverage ( gpucast_bp_trimdata, 
                                        gpucast_bp_celldata, 
                                        gpucast_bp_curvelist, 
                                        gpucast_bp_curvedata, 
                                        gpucast_preclassification,
                                        gpucast_prefilter,
                                        uv_nurbs, 
                                        duv_dx,
                                        duv_dy,
                                        trim_index, 
                                        trim_type, 
                                        tmp,
                                        gpucast_trimming_error_tolerance, 
                                        gpucast_trimming_max_bisections,
                                        GPUCAST_TRIMMING_COVERAGE_ESTIMATION);
  }

  if (gpucast_trimming_method == 2)
  {
    coverage= trimming_contour_double_binary_coverage ( gpucast_cmb_partition, 
                                                gpucast_cmb_contourlist,
                                                gpucast_cmb_curvelist,
                                                gpucast_cmb_curvedata,
                                                gpucast_cmb_pointdata,
                                                gpucast_preclassification,
                                                gpucast_prefilter,
                                                uv_nurbs, 
                                                duv_dx,
                                                duv_dy,
                                                trim_index,
                                                trim_type, 
                                                tmp,
                                                gpucast_trimming_error_tolerance, 
                                                gpucast_trimming_max_bisections,
                                                GPUCAST_TRIMMING_COVERAGE_ESTIMATION);
  }

  if (gpucast_trimming_method == 3) 
  {
    coverage = trimming_contour_kd_coverage(gpucast_kd_partition,
                                            gpucast_kd_contourlist,
                                            gpucast_kd_curvelist,
                                            gpucast_kd_curvedata,
                                            gpucast_kd_pointdata,
                                            gpucast_preclassification,
                                            gpucast_prefilter,
                                            uv_nurbs,
                                            duv_dx,
                                            duv_dy,
                                            trim_index,
                                            trim_type,
                                            tmp,
                                            gpucast_trimming_error_tolerance, 
                                            gpucast_trimming_max_bisections,
                                            GPUCAST_TRIMMING_COVERAGE_ESTIMATION);
  }

  
  if (gpucast_trimming_method == 4) {
    coverage = trimming_loop_list_coverage(uv_nurbs, duv_dx, duv_dy, gpucast_preclassification, gpucast_prefilter, trim_index, GPUCAST_TRIMMING_COVERAGE_ESTIMATION);
  }

  if ( coverage <= 0.0 )
  {
    discard;
  }

  if (gpucast_trimming_method == 0)
  {
    is_trimmed = false;
  }

  vec4 normal_world     = gpucast_normal_matrix * vec4(geometry_normal, 0.0);
  vec4 viewer           = gpucast_view_inverse_matrix * vec4(0.0, 0.0, 0.0, 1.0);

#if 1
  vec3 uv_color = vec3((uv_nurbs - nurbs_domain.xy) / domain_size, 0.0);
  out_color = shade_phong_fresnel(vec4(geometry_world_position, 1.0), 
                                 normalize(normal_world.xyz), 
                                 normalize(viewer.xyz),
                                 vec4(0.0, 0.0, 10000.0, 1.0),
                                 gpucast_material_ambient.rgb, 
                                 gpucast_material_diffuse.rgb, 
                                 gpucast_material_specular.rgb,
                                 gpucast_shininess,
                                 gpucast_opacity,
                                 bool(spheremapping),
                                 spheremap,
                                 bool(diffusemapping),
                                 diffusemap);


  submit_fragment(gl_FragCoord.z,
                  coverage * gpucast_opacity,
                  gpucast_depth_buffer, 
                  1.0, 1.0, 1.0,  // pbr
                  out_color.rgb,  
                  normal_world.rgb, 
                  false);

#else
  out_color = vec4(vec2(1.0) - uv, 0.0, coverage);
#endif

#endif

  
  /////////////////////////////////////////////////////////////////////////////
  // multisampling coverage estimation
  /////////////////////////////////////////////////////////////////////////////
#if GPUCAST_ANTI_ALIASING_MODE > 1 && GPUCAST_ANTI_ALIASING_MODE < 6
  
#if GPUCAST_MAP_BEZIERCOORDS_TO_TESSELATION
  vec2 uv           = geometry_texcoords.xy;
  vec2 uv_nurbs     = uv * domain_size + nurbs_domain.xy;

  vec2 duv_dx       = dFdx(geometry_texcoords.xy);
  vec2 duv_dy       = dFdy(geometry_texcoords.xy);
  
  duv_dx           *= domain_size;
  duv_dy           *= domain_size;
#else 
  vec2 uv_nurbs     = geometry_texcoords.xy;
  vec2 duv_dx       = dFdx(uv_nurbs);
  vec2 duv_dy       = dFdy(uv_nurbs);
#endif

  
  int trim_type = retrieve_trim_type(int(gIndex));
  int tmp = 0;

  float coverage = 1.0;

  int samples_trimmed = 0;
  int samples_total = GPUCAST_ANTI_ALIASING_MODE;
  if (samples_total == 5) samples_total = 8;

  for (int i = 0; i != samples_total; ++i) 
  {
    for (int j = 0; j != samples_total; ++j) 
    {
      vec2 uv_min = uv_nurbs - duv_dx/2 - duv_dy/2;
      vec2 uv_sample = uv_min + float(i+1)/float(samples_total+1) * duv_dx + float(j+1)/float(samples_total+1) * duv_dy;

      bool is_trimmed = false;
      if (gpucast_trimming_method == 0)
      {
        samples_trimmed += 1;
      }

      if (gpucast_trimming_method == 1)
      {
        is_trimmed = trimming_double_binary ( gpucast_bp_trimdata, 
                                              gpucast_bp_celldata, 
                                              gpucast_bp_curvelist, 
                                              gpucast_bp_curvedata, 
                                              gpucast_preclassification,
                                              uv_sample, 
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
                                                      uv_sample, 
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
                                          uv_sample,
                                          trim_index,
                                          trim_type,
                                          tmp,
                                          gpucast_trimming_error_tolerance, 
                                          gpucast_trimming_max_bisections);
      }

      if (gpucast_trimming_method == 4) {
        is_trimmed = trimming_loop_list(uv_sample, trim_index, gpucast_preclassification);
      }

      samples_trimmed += int(is_trimmed);
    }
  }

  coverage = 1.0 - float(samples_trimmed) / float(samples_total * samples_total);

  if ( coverage <= 0.0 )
  {
    discard;
  }

  vec4 normal_world     = gpucast_normal_matrix * vec4(geometry_normal, 0.0);
  vec4 viewer           = gpucast_view_inverse_matrix * vec4(0.0, 0.0, 0.0, 1.0);
  
#if 1
  out_color = shade_phong_fresnel(vec4(geometry_world_position, 1.0), 
                                 normalize(normal_world.xyz), 
                                 normalize(viewer.xyz),
                                 vec4(0.0, 0.0, 10000.0, 1.0),
                                 gpucast_material_ambient.rgb, 
                                 gpucast_material_diffuse.rgb, 
                                 gpucast_material_specular.rgb,
                                 gpucast_shininess,
                                 gpucast_opacity,
                                 bool(spheremapping),
                                 spheremap,
                                 bool(diffusemapping),
                                 diffusemap);

  submit_fragment(gl_FragDepth,
                coverage * gpucast_opacity,
                gpucast_depth_buffer, 
                1.0, 1.0, 1.0,  // pbr
                out_color.rgb,  
                normal_world.rgb, 
                false);
#else
  out_color = vec4(vec2(1.0) - uv, 0.0, coverage);
#endif

#endif

#if GPUCAST_WRITE_DEBUG_COUNTER
  if ( !is_trimmed ) {
    atomicCounterIncrement(fragment_counter);
  } else {
    atomicCounterIncrement(trimmed_fragments_counter);
  }
#endif


}