/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : renderconfig.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_RENDERCONFIG_HPP
#define GPUCAST_RENDERCONFIG_HPP

// header, system
#include <memory>

// header, project
#include <gpucast/math/oriented_boundingbox.hpp>

namespace gpucast {

struct renderconfig
{
   float      newton_epsilon;
   unsigned   newton_iterations;
   int        width;
   int        height;
   float      isovalue;
   float      steplength_min;
   float      steplength_max;
   unsigned   volume_info_offset;
   unsigned   max_binary_steps;
   bool       screenspace_newton_error;
   float      boundary_opacity;
   float      isosurface_opacity;
   float      nearplane;
   float      farplane;
   bool       show_isosides;
   float      attrib_min;
   float      attrib_max;
   bool       adaptive_sampling;
   float      steplength_scale;
   unsigned   max_octree_depth;
   float      bbox_min[3];
   float      bbox_max[3];
   bool       backface_culling;
};

struct bufferinfo 
{
  unsigned gridbuffer_size;
  unsigned octree_size;

  unsigned facebuffer_size;
  unsigned bboxbuffer_size;
  unsigned limitbuffer_size;

  unsigned surfacedata_size;
  unsigned surfacepoints_size;

  unsigned volumedata_size;
  unsigned volumepoints_size;

  unsigned attributedata_size;
  unsigned attributepoints_size;
};

} // namespace gpucast

#endif // GPUCAST_RENDERCONFIG_HPP
