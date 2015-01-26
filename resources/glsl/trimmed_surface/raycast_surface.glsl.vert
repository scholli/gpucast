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

// attributes
layout (location = 0) in vec4 vertex;
layout (location = 1) in vec4 texcoord;
layout (location = 2) in vec4 vattrib0;
layout (location = 3) in vec4 vattrib1;

// varying variables
out vec4 v_modelcoord;
out vec4 frag_texcoord;

flat out int trim_index_db;
flat out int trim_index_cmb;
flat out int data_index;
flat out int order_u;
flat out int order_v;

flat out vec4 uvrange;
flat out int  trimtype;

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

  trim_index_db  = int(floatBitsToUint(vattrib0[0]));
  trim_index_cmb = int(floatBitsToUint(vattrib0[3]));
  data_index     = int(floatBitsToUint(vattrib0[1]));
  order_u        = int(texcoord[2]);
  order_v        = int(texcoord[3]);

  uvrange        = vattrib1;
  trimtype       = int(floatBitsToUint(vattrib0[2]));

  /* transform convex hull in modelview to generate fragments */
  gl_Position = modelviewprojectionmatrix * vertex;
}
