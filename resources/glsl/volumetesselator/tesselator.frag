/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : tesselator.frag
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
uniform vec4          global_attribute_min;
uniform vec4          global_attribute_max;

uniform vec4          iso_threshold;

uniform mat4          modelviewprojectionmatrix;
uniform mat4          modelviewmatrix;
uniform mat4          modelviewmatrixinverse;
uniform mat4          normalmatrix;

uniform sampler2D     transfertexture;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragcolor;
in vec4 fragposition;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 out_color;

/********************************************************************************
* functions
********************************************************************************/
#include "./libgpucast/glsl/isosurface/target_function.frag"
#include "./libgpucast/glsl/base/faceforward.frag"


///////////////////////////////////////////////////////////////////////////////
vec4 phong_shading ( in vec4 point,
                     in vec4 normal,
                     in vec4 lightpos )
{
  vec3 L = normalize ( lightpos.xyz - point.xyz );
  vec3 N = normalize ( normal.xyz );

  N = faceforward( -point.xyz, N ); // faceforward doesn't work properly

  float diffuse = max( 0.0, dot (N , L));
  vec4  color   = vec4(diffuse, diffuse, diffuse, 1.0);
  return color;
}


/********************************************************************************
* shader for raycasting a beziervolume
********************************************************************************/
void main(void)
{
  const float epsilon = 0.001;

  vec4 dx = dFdx(fragposition);
  vec4 dy = dFdy(fragposition);

  vec3 normal = cross( normalize(dx.xyz), normalize(dy.xyz) );
  normal = faceforward ( -fragposition.xyz, normal );

  out_color = vec4((fragcolor.xyz - global_attribute_min.xyz) / (global_attribute_max.xyz - global_attribute_min.xyz), 1.0);
  out_color *= phong_shading ( fragposition, vec4(normal,0.0), vec4(0.0, 0.0, 0.0, 1.0));

  out_color.w = 1.0;
}

