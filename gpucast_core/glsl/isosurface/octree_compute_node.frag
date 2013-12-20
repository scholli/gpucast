/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : octree_compute_ocnode.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

void compute_ocnode ( in vec3  sample_point,
                      in vec3  treebounds_min,
                      in vec3  treebounds_max,
                      in int   depth,
                      out vec3 ocnode_bbox_min,
                      out vec3 ocnode_bbox_max )
{
  vec3 treebounds = treebounds_max - treebounds_min;

  float scale       = pow(2.0, float(depth));
  vec3 local_coords = (sample_point - treebounds_min) / treebounds;
  ocnode_bbox_min   = floor (local_coords * scale) / scale;
  ocnode_bbox_max   = ceil  (local_coords * scale) / scale;

  ocnode_bbox_min  *= treebounds;
  ocnode_bbox_max  *= treebounds;

  ocnode_bbox_min  += treebounds_min;
  ocnode_bbox_max  += treebounds_min;
}
