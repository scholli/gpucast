/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : cuda_globals.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_RENDERCONFIG_H
#define LIBGPUCAST_CUDA_RENDERCONFIG_H

#include <gpucast/volume/isosurface/renderconfig.hpp>

#if 0
__device__
struct cuda_renderconfig
{
  cuda_renderconfig ( gpucast::renderconfig const& r )
    : width                    ( r.width                    )
      height                   ( r.height                   )
      backface_culling         ( r.backface_culling         )
      nearplane                ( r.nearplane                )
      farplane                 ( r.farplane                 )
      show_isosides            ( r.show_isosides            )
      boundary_opacity         ( r.boundary_opacity         )
      isosurface_opacity       ( r.isosurface_opacity       )
      newton_epsilon           ( r.newton_epsilon           )
      newton_iterations        ( r.newton_iterations        )
      steplength_min           ( r.steplength_min           )
      steplength_max           ( r.steplength_max           )
      steplength_scale         ( r.steplength_scale         )
      max_binary_steps         ( r.max_binary_steps         )
      volume_info_offset       ( r.volume_info_offset       )
      screenspace_newton_error ( r.screenspace_newton_error )
      adaptive_sampling        ( r.adaptive_sampling        )
      isovalue                 ( r.isovalue                 )
      octree_min               ( r.octree_min               )
      octree_max               ( r.octree_max               )
      attrib_min               ( r.attrib_min               )
      attrib_max               ( r.attrib_max               )
  {}                           

  // renderer state configuration
  unsigned    width;
  unsigned    height;
  bool        backface_culling;
  float       nearplane;
  float       farplane;

  // renderer visual appearance
  bool        show_isosides;
  float       boundary_opacity;
  float       isosurface_opacity;

  // isosurface raytracing configuration
  float       newton_epsilon;
  unsigned    newton_iterations;

  float       steplength_min;
  float       steplength_max;
  float       steplength_scale;

  unsigned    max_binary_steps;
  unsigned    volume_info_offset;

  bool        screenspace_newton_error;
  bool        adaptive_sampling;

  // data dependent values
  float       isovalue;

  float3      octree_min;
  float3      octree_max;

  float       attrib_min;
  float       attrib_max;
};
#endif

#endif