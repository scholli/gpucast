/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : draw_from_textures.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

/********************************************************************************
* constants
********************************************************************************/

/********************************************************************************
* uniforms
********************************************************************************/
// screen and camera setup 
uniform int           width;
uniform int           height;

uniform sampler2D     colortexture;
uniform sampler2D     depthtexture;

/********************************************************************************
* input
********************************************************************************/
in vec4 fragment_texcoord;

/********************************************************************************
* output
********************************************************************************/
layout (location = 0) out vec4  out_color;
//layout (depth_less)   out float gl_FragDepth;

/********************************************************************************
* functions
********************************************************************************/
vec4 filter_pixel_error ( sampler2D color,
                          vec2      coord,
                          float     tolerance )
{
  vec4 n  = texture(colortexture, fragment_texcoord.xy );
  vec4 n0 = texture(colortexture, fragment_texcoord.xy + vec2( 0.0f/width, 1.0/height) );
  vec4 n1 = texture(colortexture, fragment_texcoord.xy + vec2( 1.0f/width, 0.0/height) );
  vec4 n2 = texture(colortexture, fragment_texcoord.xy + vec2( 1.0f/width, 1.0/height) );
  vec4 n3 = texture(colortexture, fragment_texcoord.xy + vec2( 0.0f/width,-1.0/height) );
  vec4 n4 = texture(colortexture, fragment_texcoord.xy + vec2(-1.0f/width, 0.0/height) );
  vec4 n5 = texture(colortexture, fragment_texcoord.xy + vec2(-1.0f/width,-1.0/height) );
  vec4 n6 = texture(colortexture, fragment_texcoord.xy + vec2( 1.0f/width,-1.0/height) );
  vec4 n7 = texture(colortexture, fragment_texcoord.xy + vec2(-1.0f/width, 1.0/height) );

  vec4 avg = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7) / 8.0;

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
* raycast pass traverses intervals per fragment and intersects surfaces
********************************************************************************/
void main(void)
{
  //out_color               = texture(colortexture, fragment_texcoord.xy);

  float depth = texture ( depthtexture, fragment_texcoord.xy).x;

  if ( depth == 1.0 ) {
    //discard;
  } else {
    //gl_FragDepth  = depth;
    out_color = filter_pixel_error ( colortexture, fragment_texcoord.xy, 0.1 );
  }
}
