/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : parameter_on_boundary.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

///////////////////////////////////////////////////////////////////////////////
bool parameter_on_domain_boundary ( in vec3 uvw, 
                                    in vec3 uvwmin_local,
                                    in vec3 uvwmax_local,
                                    in vec3 uvwmin_global,
                                    in vec3 uvwmax_global,
                                    in float epsilon )
{
  vec3 uvw_global = uvwmin_local + uvw * (uvwmax_local - uvwmin_local);

  return uvw_global.x > uvwmax_global.x - epsilon ||
         uvw_global.y > uvwmax_global.y - epsilon ||
         uvw_global.z > uvwmax_global.z - epsilon ||
         uvw_global.x < uvwmin_global.x + epsilon ||
         uvw_global.y < uvwmin_global.y + epsilon ||
         uvw_global.z < uvwmin_global.z + epsilon;
}

