/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : raycast_volume.glsl.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* constants
********************************************************************************/


/********************************************************************************
* uniforms
********************************************************************************/
uniform samplerBuffer databuffer;
uniform samplerBuffer attributebuffer;

uniform mat4          modelviewprojectionmatrix;
uniform mat4          modelviewmatrix;
uniform mat4          modelviewmatrixinverse;
uniform mat4          normalmatrix;

uniform float         nearplane;
uniform float         farplane;
uniform int           newton_iterations;
uniform float         newton_epsilon;

uniform vec4          global_attribute_min;
uniform vec4          global_attribute_max;

// isosurface rendering
uniform float         threshold;
uniform int           volume_info_offset;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragposition;

flat in int surface_id;
flat in int volume_info_id;

flat in int s_id;
flat in int t_id;
flat in int u_id;

in vec3 parameter;


/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 out_color;


/********************************************************************************
* functions
********************************************************************************/
#include "./libgpucast/glsl/base/compute_depth.frag"

#include "./libgpucast/glsl/math/euclidian_space.glsl.frag"
#include "./libgpucast/glsl/math/adjoint.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface.glsl.frag"
#include "./libgpucast/glsl/math/horner_surface_derivatives.glsl.frag"
#include "./libgpucast/glsl/math/horner_volume.glsl.frag"
#include "./libgpucast/glsl/math/raygeneration.glsl.frag"
#include "./libgpucast/glsl/math/newton_surface.glsl.frag"
#include "./libgpucast/glsl/math/newton_volume.glsl.frag"


/********************************************************************************
* shader for raycasting a beziervolume
********************************************************************************/
void main(void)
{
  /*********************************************************************
  * retrieve data indices
  *********************************************************************/
  vec4 volume_info      = texelFetchBuffer(databuffer, volume_info_id);
  vec4 volume_info2     = texelFetchBuffer(databuffer, volume_info_id + 1);

  const int base_offset = 1;
  int volume_id         = int(volume_info.x); 
  int surface_base_id   = int(volume_info.y);
  int attribute_id      = int(volume_info.z);

  ivec3 order           = ivec3(volume_info2.x, volume_info2.y, volume_info2.z);

  /*********************************************************************
  * Ray generation
  *********************************************************************/
  vec3 n1, n2;
  float d1, d2;
  raygen(fragposition, modelviewmatrixinverse, n1, n2, d1, d2);

  /*********************************************************************
  * Surface intersection
  *********************************************************************/
  vec4 p      = vec4(0.0);
  vec4 du     = vec4(0.0);
  vec4 dv     = vec4(0.0);

  vec2 uv     = vec2(parameter[s_id], parameter[t_id]);

  // intersect ray with surface
  bool surface_intersection = newton(uv, newton_epsilon, newton_iterations, databuffer, surface_id, order[s_id], order[t_id], n1, n2, d1, d2, p, du, dv);
  vec3 n                    = cross(normalize(du.xyz), normalize(dv.xyz));

  if (!surface_intersection) 
  {
    discard;
  }

  // compute volumetric uvw parameter
  vec3 uvw  = vec3(0.0);
  uvw[s_id] = uv.x;
  uvw[t_id] = uv.y;
  uvw[u_id] = clamp(parameter[u_id], 0.0, 1.0);

  /*********************************************************************
   * depth correction
   *********************************************************************/
  vec4 pview = modelviewmatrix * vec4(p.xyz, 1.0);  
  gl_FragDepth = compute_depth ( pview, nearplane, farplane );

  /*********************************************************************
  * shade
  *********************************************************************/
  int iters = 0;
  bool hit = false;
  vec4 dp, ddu, ddv, ddw;

  // do initial evaluation at starting point to determine iteration direction
  //vec4 attribute_min = texelFetchBuffer(attributebuffer, attribute_id    );
  //vec4 attribute_max = texelFetchBuffer(attributebuffer, attribute_id + 1);

  evaluateVolume(attributebuffer, attribute_id + volume_info_offset, order.x, order.y, order.z, uvw.x, uvw.y, uvw.z, dp, ddu, ddv, ddw);  

  vec4 lightpos   = vec4 ( 0.0, 0.0, 0.0, 1.0); // light from camera

  vec3 L          = normalize ( lightpos.xyz - pview.xyz );
  vec3 N          = normalize ( (normalmatrix * vec4(n, 0.0)).xyz );

  if ( dot (N, -pview.xyz) < -0.0)
  {
    N = -N;
  }

  float diffuse   = max( 0.0, dot (N , L));
  vec4 rel_attrib = vec4((dp.xyz - global_attribute_min.xyz) / (global_attribute_max.xyz - global_attribute_min.xyz), 1.0);

  out_color       = vec4(0.0, 0.0, 0.0, 1.0);
  out_color       = diffuse * vec4(rel_attrib.xyz, 1.0);
  //out_color       = clamp ( out_color, vec4(0.0), vec4(1.0));
  //out_color       = vec4(uvw, 1.0);
  //out_color       = vec4(uv, 0.0, 1.0);

  out_color.w     = 1.0;
}

