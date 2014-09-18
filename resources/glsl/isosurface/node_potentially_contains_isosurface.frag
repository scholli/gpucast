/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : node_potentially_contains_isosurface.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

///////////////////////////////////////////////////////////////////////////////
bool node_potentially_contains_isosurface ( in samplerBuffer limitbuffer,
                                            in ivec4         node, 
                                            in vec4          ray_entry, 
                                            in vec4          ray_direction, 
                                            in vec4          octree_min, 
                                            in vec4          octree_max,
                                            in vec4          isovalue)
{
  // index for limitbuffer
  int limit_id  = node.y;

  // read limits of ocnode from texture
  vec4 node_min = texelFetchBuffer(limitbuffer, limit_id    );
  vec4 node_max = texelFetchBuffer(limitbuffer, limit_id + 1);

  // check if iso_value is in range of limits
  bool threshold_in_interval = is_in_range (isovalue, node_min, node_max);

  return threshold_in_interval;
}