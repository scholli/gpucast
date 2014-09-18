/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : copy_depth.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#version 420 core
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_draw_buffers : enable

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
uniform sampler2D depth_texture;
uniform sampler2D color_texture;

uniform int       width;
uniform int       height;

/********************************************************************************
* input
********************************************************************************/
in vec4 frag_texcoord;

/********************************************************************************
* output
********************************************************************************/
layout (location=0) out vec4 out_data;

/********************************************************************************
* functions
********************************************************************************/
vec4 filter_pixel_error ( sampler2D color,
                          vec2      coord,
                          float     tolerance )
{
  vec4 n  = texture(color, coord.xy );
  vec4 n0 = texture(color, coord.xy + vec2( 0.0f/width, 1.0/height) );
  vec4 n1 = texture(color, coord.xy + vec2( 1.0f/width, 0.0/height) );
  vec4 n2 = texture(color, coord.xy + vec2( 1.0f/width, 1.0/height) );
  vec4 n3 = texture(color, coord.xy + vec2( 0.0f/width,-1.0/height) );
  vec4 n4 = texture(color, coord.xy + vec2(-1.0f/width, 0.0/height) );
  vec4 n5 = texture(color, coord.xy + vec2(-1.0f/width,-1.0/height) );
  vec4 n6 = texture(color, coord.xy + vec2( 1.0f/width,-1.0/height) );
  vec4 n7 = texture(color, coord.xy + vec2(-1.0f/width, 1.0/height) );

  vec4 avg = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7) / 8.0f;

  float max_error = 1.0;

  max_error = min ( length ( abs ( n0 - n ) ), max_error );
  max_error = min ( length ( abs ( n1 - n ) ), max_error );
  max_error = min ( length ( abs ( n2 - n ) ), max_error );
  max_error = min ( length ( abs ( n3 - n ) ), max_error );
  max_error = min ( length ( abs ( n4 - n ) ), max_error );
  max_error = min ( length ( abs ( n5 - n ) ), max_error );
  max_error = min ( length ( abs ( n6 - n ) ), max_error );
  max_error = min ( length ( abs ( n7 - n ) ), max_error );

  if ( max_error > tolerance)
  {
    return avg;
  } else {
    return n;
  }
}

/********************************************************************************
* copy depth texture to color attachment
********************************************************************************/
void main(void)
{
  vec3 color  = texture ( color_texture, frag_texcoord.xy).xyz;
  float depth = texture ( depth_texture, frag_texcoord.xy).x;
  out_data    = vec4 ( color, depth );
}

