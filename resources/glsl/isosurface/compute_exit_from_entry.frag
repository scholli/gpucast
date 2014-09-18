/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_exit_from_entry.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

vec4 compute_exit_from_entry ( in vec4 entry_point,
                               in vec4 entry_direction,
                               in vec3 bbox_min,
                               in vec3 bbox_max,
                               out float t,
                               out vec3 exit_normal)
{
  t           = 0;
  exit_normal = vec3(0.0);
  float tx    = 0.0;
  float ty    = 0.0;
  float tz    = 0.0;

  tx = max((bbox_max.x - entry_point.x) / entry_direction.x, (bbox_min.x - entry_point.x) / entry_direction.x);
  ty = max((bbox_max.y - entry_point.y) / entry_direction.y, (bbox_min.y - entry_point.y) / entry_direction.y);
  tz = max((bbox_max.z - entry_point.z) / entry_direction.z, (bbox_min.z - entry_point.z) / entry_direction.z);

  if ( tx <= ty && tx <= tz ) 
  {
    t = tx; 
    exit_normal = vec3(sign(entry_direction.x), 0.0, 0.0);
    return entry_point + t * entry_direction;
  }

  if ( ty <= tx && ty <= tz ) 
  {
    t = ty; 
    exit_normal = vec3(0.0, sign(entry_direction.y), 0.0);
    return entry_point + t * entry_direction;
  }

  if ( tz <= tx && tz <= ty ) 
  {
    t = tz; 
    exit_normal = vec3(0.0, 0.0, sign(entry_direction.z));
    return entry_point + t * entry_direction;
  }
}
