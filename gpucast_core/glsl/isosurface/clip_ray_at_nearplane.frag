/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : clip_ray_at_nearplane.frag
*  project    : gpucast
*  description:
*
********************************************************************************/


/////////////////////////////////////////////////////////////////////////////
vec4 clip_ray_at_nearplane (in mat4   modelview,
                            in mat4   modelview_inverse,
                            in vec4   ray_origin, 
                            in vec4   ray_direction, 
                            in float  near )
{
  vec4 ray_entry_world     = modelview * ray_origin;
  vec4 ray_direction_world = modelview * ray_direction;

  if ( ray_entry_world.z > -near )
  {
    float z_offset = (near - ray_entry_world.z) / ray_direction_world.z;
    return vec4(modelview_inverse * (ray_entry_world + z_offset * ray_direction_world));
  } else {
    return ray_origin;
  }
}


