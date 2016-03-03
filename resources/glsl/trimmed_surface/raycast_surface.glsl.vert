 /********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : raycast_surface.glsl.vert                             
*  project    : gpucast 
*  description: 
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable

#include "./resources/glsl/common/conversion.glsl"

// attributes
layout (location = 0) in vec4 vertex;
layout (location = 1) in vec4 texcoord;
layout (location = 2) in vec4 vattrib2;
layout (location = 3) in vec4 vattrib3;

// varying variables
out vec4 v_modelcoord;
out vec4 frag_texcoord;

flat out int trim_index;
flat out int trim_approach;
flat out int trim_type;

flat out int obb_index;
flat out int data_index;
flat out int order_u;
flat out int order_v;

flat out vec4 uvrange;


// uniforms
uniform mat4 modelviewmatrix;
uniform mat4 modelviewprojectionmatrix;
uniform mat4 normalmatrix;
uniform mat4 modelviewmatrixinverse;


void main(void)
{
  /* rasterize model coordinates and TexCoord */
  v_modelcoord   = vertex;
  frag_texcoord  = texcoord;

  trim_index     = int(floatBitsToUint(vattrib2[0]));

  uvec2 type_approach = intToUInt2(floatBitsToUint(vattrib2[2]));
  trim_type      = int(type_approach.x);
  trim_approach  = int(type_approach.y);

  obb_index      = int(floatBitsToUint(vattrib2[3]));
  data_index     = int(floatBitsToUint(vattrib2[1]));
  order_u        = int(floatBitsToUint(texcoord[2]));
  order_v        = int(floatBitsToUint(texcoord[3]));

  uvrange        = vattrib3;

  /* transform convex hull in modelview to generate fragments */
  gl_Position = modelviewprojectionmatrix * vertex;
}
