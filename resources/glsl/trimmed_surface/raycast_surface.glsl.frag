/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : raycast_surface.glsl.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_separate_shader_objects : enable 

/****************************************************
* choose type of root finder for trimming
****************************************************/
precision highp float;

/*******************************************************************************
 *   vertexdata : control points in model coordinates
 *   trimdata   : trim curve acceleration structure
 *   urangeslist: u-intervals per v-interval
 *   curvelist  : curve lists according to a certain range [vmin vmax]
 *   curvedata  : control points of trim curves
 *
 *   trim_index : start index in trim acceleration for current surface
 *   data_index : start index in control point buffer for current surface
 *   uvrange    : parameter range of surface [umin, umax, vmin, vmax]
 *   trimtype   : 0 = inner is trimmed, 1 outer is trimmed
 ******************************************************************************/

uniform samplerBuffer vertexdata;

uniform samplerBuffer cmb_partition;
uniform samplerBuffer cmb_contourlist;
uniform samplerBuffer cmb_curvelist;
uniform samplerBuffer cmb_curvedata;
uniform samplerBuffer cmb_pointdata;

uniform samplerBuffer bp_trimdata;
uniform samplerBuffer bp_celldata;
uniform samplerBuffer bp_curvelist;
uniform samplerBuffer bp_curvedata;

/*******************************************************************************
 *  Trimming for parameter pair [u,v] : DATA STRUCTURE :
 *
 *  - trimdata :
 *      [index]
 *      trim_index          : [vmin_all, vmax_all, #tdata, 0.0]
 *      trim_index + 1      : [vmin_clist, vmax_clist, urangeslist_id, 0.0]
 *      ...                    ...
 *      trim_index + #tdata : [vmin_clist, vmax_clist, urangeslist_id, 0.0]
 *
 *  - urangeslist :
 *      [index]
 *      urangeslist_id                  : [umin_all, umax_all, nr_of_uranges, 0.0]
 *      urangeslist_id + 1              : [umin,     umax, intersect_on_right, curvelist_id]
 *             ...
 *      urangeslist_id + nr_of_uranges  : [umin,     umax, intersect_on_right, curvelist_id]
 *
 *  -curvelist :
 *      [index]
 *      curvelist_id                    : [nr_of_curves]
 *      curvelist_id + 1                : [curve_id, curveorder, increasing, 0.0]
 *      ...
 *      curvelist_id + nr_of_curves     : [curve_id, curveorder, increasing, 0.0]
 *
 *  - curvedata :
 *      [index]
 *      curve_id            : [x0, y0, w0, 0.0]
 *             ...
 *      curve_id + order    : [xn, yn, wn, 0.0]
 *******************************************************************************/

/*******************************************************************************
 * cubemap for reflections
 ******************************************************************************/
uniform sampler2D   spheremap;
uniform sampler2D   diffusemap;

/*******************************************************************************
 * VARYINGS :
 *   v_modelcoord : rasterized model coordinates
 ******************************************************************************/
in vec4 v_modelcoord;
in vec4 frag_texcoord;

flat in int trim_index_db;
flat in int trim_index_cmb;
flat in int data_index;
flat in int order_u;
flat in int order_v;

flat in vec4 uvrange;
flat in int  trimtype;

/*******************************************************************************
 * UNIFORMS :
 ******************************************************************************/

/*uniform int trim_iterations;*/
uniform int   iterations;
uniform float epsilon_object_space;
uniform float nearplane;
uniform float farplane;
uniform int   trimapproach;

uniform int   spheremapping;
uniform int   diffusemapping;

uniform int   trimming_enabled;
uniform int   raycasting_enabled;

/* material definition */
uniform vec3  mat_ambient;
uniform vec3  mat_diffuse;
uniform vec3  mat_specular;
uniform float shininess;
uniform float opacity;

uniform mat4  modelviewmatrix;
uniform mat4  modelviewprojectionmatrix;
uniform mat4  normalmatrix;
uniform mat4  modelviewmatrixinverse;

uniform vec4  lightpos0;

/*******************************************************************************
 * OUTPUT
 ******************************************************************************/
layout (location = 0) out vec4  out_color;
layout (depth_any)    out float gl_FragDepth;

/*******************************************************************************
 * include functions
 ******************************************************************************/
#include "./resources/glsl/base/compute_depth.frag"
#include "./resources/glsl/base/conversion.glsl"
#include "./resources/glsl/math/adjoint.glsl.frag"
#include "./resources/glsl/math/euclidian_space.glsl.frag"
#include "./resources/glsl/math/horner_surface.glsl.frag"
#include "./resources/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./resources/glsl/math/horner_curve.glsl"
#include "./resources/glsl/math/newton_surface.glsl.frag"
#include "./resources/glsl/math/raygeneration.glsl.frag" 
#include "./resources/glsl/trimming/trimming_contour_double_binary.glsl"
#include "./resources/glsl/trimming/trimming_double_binary.glsl"
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
  raygen(v_modelcoord, modelviewmatrixinverse, n1, n2, d1, d2);

  /*********************************************************************
  * Surface intersection
  *********************************************************************/
  vec2 uv = vec2(frag_texcoord[0], frag_texcoord[1]);
  
  vec4 p  = vec4(0.0);
  vec4 du = vec4(0.0);
  vec4 dv = vec4(0.0);

  bool surface_hit = true;
  //if ( bool(raycasting_enabled) )
  {
    // raycast for ray surface intersection
    surface_hit = newton(uv, 0.001f, iterations, vertexdata, data_index, order_u, order_v, n1, n2, d1, d2, p, du, dv);
  }

  if ( !surface_hit )//&& bool(raycasting_enabled) )
  {
    discard;
  }
  

  if ( !bool(raycasting_enabled) )
  {
    p = v_modelcoord;
  }

  vec3 normal = normalize(cross(normalize(du.xyz), normalize(dv.xyz)));
  
  /*********************************************************************
   * Trimming process
   *********************************************************************/
  
  // transform in NURBS parameter coordinates
  uv[0] = uvrange[0] + uv[0] * (uvrange[1] - uvrange[0]);
  uv[1] = uvrange[2] + uv[1] * (uvrange[3] - uvrange[2]);
  int iters = 0;
  vec4 debug = vec4(0.0);

  bool is_trimmed = false;


  //if ( trimapproach == 0 )
  //{ 
  //  is_trimmed = trimming_double_binary ( bp_trimdata, 
  //                                        bp_celldata, 
  //                                        bp_curvelist, 
  //                                        bp_curvedata, 
  //                                        uv, 
  //                                        trim_index_db, 
  //                                        trimtype, 
  //                                        iters, 
  //                                        0.00001, 
  //                                        16 );
  //} 
  //
  //if ( trimapproach == 1 )
  //{
    is_trimmed= trimming_contour_double_binary ( cmb_partition, 
                                                 cmb_contourlist,
                                                 cmb_curvelist,
                                                 cmb_curvedata,
                                                 cmb_pointdata,
                                                 uv, 
                                                 trim_index_cmb, 
                                                 trimtype, 
                                                 iters, 
                                                 0.00001, 
                                                 16 );
  //}

  if ( bool(trimming_enabled) && is_trimmed)
  {
    discard;
  }

  /*********************************************************************
   * depth correction
   *********************************************************************/
  vec4 p_world = modelviewmatrix * vec4(p.xyz, 1.0);
  float corrected_depth = compute_depth ( p_world, nearplane, farplane );
  gl_FragDepth = corrected_depth;

  /*********************************************************************
   * Shading process
   ********************************************************************/
  if (bool(raycasting_enabled)) 
  {
    out_color = shade_phong_fresnel(p_world, 
                                    normalize((normalmatrix * vec4(normal, 0.0)).xyz), 
                                    vec4(1.0, 1.0, 1.0, 1.0),
                                    mat_ambient, mat_diffuse, mat_specular,
                                    shininess,
                                    opacity,
                                    bool(spheremapping),
                                    spheremap,
                                    bool(diffusemapping),
                                    diffusemap);
  } else {
    out_color = vec4(frag_texcoord.xy, 0.0, 1.0);
  }
}



