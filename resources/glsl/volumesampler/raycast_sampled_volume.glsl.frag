/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : raycast_sampled_volume.glsl.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
uniform vec4 attribute_min;
uniform vec4 attribute_max;

uniform vec4 isovalue;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragcolor;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4 color;

/********************************************************************************
* functions
********************************************************************************/
float target_function( in vec4 data )
{
  return data[0];
}

/********************************************************************************
* shader for raycasting a beziervolume
********************************************************************************/
void main(void)
{
  const float epsilon = 0.001;

  //if ( target_function ( fragcolor ) + epsilon > target_function ( isovalue ) &&
  //     target_function ( fragcolor ) - epsilon < target_function ( isovalue )  )
  {
    color = vec4((fragcolor.xyz - attribute_min.xyz) / (attribute_max.xyz - attribute_min.xyz), 1.0);
  //} else {
    //color = vec4((fragcolor.xyz - attribute_min.xyz) / (attribute_max.xyz - attribute_min.xyz), 1.0);
    //discard;
  }
}

