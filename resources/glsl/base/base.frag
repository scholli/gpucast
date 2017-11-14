/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : base.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#extension GL_NV_gpu_shader5 : enable

in vec4 color;

layout (location = 0) out vec4 out_color;

void main(void)
{
  out_color = color;
}

