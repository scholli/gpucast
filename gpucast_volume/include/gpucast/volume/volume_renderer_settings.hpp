/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_settings.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_SETTINGS_HPP
#define GPUCAST_VOLUME_RENDERER_SETTINGS_HPP

#include <gpucast/volume/gpucast.hpp>

#include <device_types.h>

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
struct visualization_properties 
{
  __device__ __host__ inline 
  visualization_properties() 
    : show_samples_isosurface_intersection(false),
      show_samples_face_intersection(false),
      show_face_intersections(false),
      show_face_intersection_tests(false),
      show_isosides(false)
  {
    background[0] = 1.0f;
    background[1] = 1.0f;
    background[2] = 1.0f;
  }

  bool  show_samples_isosurface_intersection;
  bool  show_samples_face_intersection;
  bool  show_face_intersections;
  bool  show_face_intersection_tests;
  bool  show_isosides;

  float background[3];
};

////////////////////////////////////////////////////////////////////////////////
struct render_settings 
{
  __device__ __host__ inline 
  render_settings()
    : width                          (1),
      height                         (1),
      fxaa                           (false),
      cullface                       (false),
      sample_based_face_intersection (false),
      detect_implicit_inflection     (false),
      detect_implicit_extremum       (false),
      newton_iterations              (5),
      newton_epsilon                 (0.001f),
      newton_screenspace_epsilon     (false),
      octree_max_depth               (16),
      octree_max_volumes_per_node    (64),
      nearplane                      (0.1f),
      farplane                       (1000.0f),
      max_bisections                 (8),
      adaptive_sampling              (true),
      global_min_attribute           (0.0f),
      global_max_attribute           (1.0f),
      relative_attribute_value       (1.0f),
      min_sample_distance            (0.01f),
      max_sample_distance            (0.2f),
      sample_step_scale              (1.0f),
      opacity_surface                (0.2f),
      opacity_isosurface             (0.4f),
      pagesize                       (4),
      allocation_grid_width          (64),
      allocation_grid_height         (64)
  {}

  unsigned                  width;
  unsigned                  height;

  bool                      fxaa;
  bool                      cullface;

  bool                      sample_based_face_intersection;
  bool                      detect_implicit_inflection;
  bool                      detect_implicit_extremum;

  unsigned                  newton_iterations;
  float                     newton_epsilon;
  bool                      newton_screenspace_epsilon;

  unsigned                  octree_max_depth;
  unsigned                  octree_max_volumes_per_node;

  float                     nearplane;
  float                     farplane;
  
  float                     global_min_attribute;
  float                     global_max_attribute;
  float                     relative_attribute_value;

  bool                      adaptive_sampling;
  unsigned                  max_bisections;
  float                     min_sample_distance;
  float                     max_sample_distance;
  float                     sample_step_scale;

  float                     opacity_surface;
  float                     opacity_isosurface;

  unsigned                  pagesize;
  int                       allocation_grid_width;
  int                       allocation_grid_height;

  visualization_properties  visualization_props;

  inline std::ostream& print (std::ostream& os) const
  {
    os << "\n fxaa : " <<                           fxaa;
    os << "\n cullface : " <<                       cullface;
    os << "\n sample_based_face_intersection : " << sample_based_face_intersection;
    os << "\n detect_implicit_inflection : " <<     detect_implicit_inflection;
    os << "\n detect_implicit_extremum : " <<       detect_implicit_extremum;
    os << "\n newton_iterations : " <<              newton_iterations;
    os << "\n newton_epsilon : " <<                 newton_epsilon;
    os << "\n newton_screenspace_epsilon : " <<     newton_screenspace_epsilon;
    os << "\n octree_max_depth : " <<               octree_max_depth;
    os << "\n octree_max_volumes_per_node : " <<    octree_max_volumes_per_node;
    os << "\n nearplane : " <<                      nearplane;
    os << "\n farplane : " <<                       farplane;
    os << "\n isosearch_max_binary_steps : " <<     max_bisections;
    os << "\n isosearch_adaptive_sampling : " <<    adaptive_sampling;
    os << "\n global_min_attribute     "   << global_min_attribute     ;
    os << "\n global_max_attribute     "   << global_max_attribute     ;
    os << "\n relative_value : " <<       relative_attribute_value;
    os << "\n min_sample_distance : " <<  min_sample_distance;
    os << "\n max_sample_distance : " <<  max_sample_distance;
    os << "\n sample_step_scale : " <<    sample_step_scale;
    os << "\n opacity_surface : " <<                opacity_surface;
    os << "\n opacity_isosurface : " <<             opacity_isosurface;
    os << "\n width                    "   << width                    ;
    os << "\n height                   "   << height                   ;
    os << "\n relative_attribute_value "   << relative_attribute_value ;
    os << "\n pagesize                 "   << pagesize                 ;
    os << "\n allocation_grid_width    "   << allocation_grid_width    ;
    os << "\n allocation_grid_height   "   << allocation_grid_height   ;

    return os;                                         
  }   
};
                                        
                                                
} // namespace gpucast
                                                
#endif // GPUCAST_VOLUME_RENDERER_SETTINGS_HPP  
