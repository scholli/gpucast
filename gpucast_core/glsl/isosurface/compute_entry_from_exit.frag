/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_entry_from_exit.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

vec4 compute_entry_from_exit ( in vec4 exit_point,
                               in vec4 exit_direction,
                               in vec3 bbox_min,
                               in vec3 bbox_max,
                               out float t,
                               out vec3 entry_normal )
{
  t = 0.0;
  entry_normal = vec3(0.0);

  float tx = max((bbox_max.x - exit_point.x) / (-exit_direction.x), (bbox_min.x - exit_point.x) / (-exit_direction.x));
  float ty = max((bbox_max.y - exit_point.y) / (-exit_direction.y), (bbox_min.y - exit_point.y) / (-exit_direction.y));
  float tz = max((bbox_max.z - exit_point.z) / (-exit_direction.z), (bbox_min.z - exit_point.z) / (-exit_direction.z));

  if ( tx < ty && tx < tz ) 
  {
    t = tx; 
    entry_normal = vec3(sign(exit_direction.x), 0.0, 0.0);
    return exit_point + t * (-exit_direction);
  }

  if ( ty < tx && ty < tz ) 
  {
    t = ty; 
    entry_normal = vec3(0.0, sign(exit_direction.y), 0.0);
    return exit_point + t * (-exit_direction);
  }

  if ( tz < tx && tz < ty ) 
  {
    t = tz; 
    entry_normal = vec3(0.0, 0.0, sign(exit_direction.z));
    return exit_point + t * (-exit_direction);
  }
}

