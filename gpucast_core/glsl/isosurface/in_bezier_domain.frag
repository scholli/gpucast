/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : in_bezier_domain.frag
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_IN_BEZIER_DOMAIN_FRAG
#define LIBGPUCAST_IN_BEZIER_DOMAIN_FRAG


/////////////////////////////////////////////////////////////////////////////
bool in_bezier_domain ( in vec3 uvw )
{
  return  uvw.x >= 0.0 && 
          uvw.y >= 0.0 && 
          uvw.z >= 0.0 && 
          uvw.x <= 1.0 && 
          uvw.y <= 1.0 && 
          uvw.z <= 1.0; 
}

/////////////////////////////////////////////////////////////////////////////
bool in_bezier_domain ( in vec2 uvw )
{
  return  uvw.x >= 0.0 && 
          uvw.y >= 0.0 && 
          uvw.x <= 1.0 && 
          uvw.y <= 1.0; 
}

#endif