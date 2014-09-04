/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : surface_renderer.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/surface_renderer.hpp"

// header, system

// header, project
#include <gpucast/core/beziersurfaceobject.hpp>


namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  surface_renderer::surface_renderer( int argc, char** argv )
    :  renderer           (),
       _cube_mapping      ( false ),
       _sphere_mapping    ( false ),
       _diffuse_mapping   ( false ),
       _trimming_enabled  ( true ),
       _raycasting_enabled( true ),
       _trimapproach      ( double_binary_partition ),
       _newton_iterations ( 5 )
  {
    _pathlist.insert(".");
    _pathlist.insert("../..");
    _pathlist.insert("../../gpucast_core/cl");
    _pathlist.insert("../../gpucast_core/glsl");
    _pathlist.insert("../../gpucast_core");
    _pathlist.insert("./cl");
    _pathlist.insert("./data/images");
    _pathlist.insert("./glsl");
    _pathlist.insert("./gpucast_core/glsl");
    _pathlist.insert("./gpucast_core/cl");
  }


  /////////////////////////////////////////////////////////////////////////////
  surface_renderer::~surface_renderer()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::newton_iterations ( std::size_t n )
  {
    _newton_iterations = n;
  }


  /////////////////////////////////////////////////////////////////////////////
  std::size_t           
  surface_renderer::newton_iterations ( ) const
  {
    return _newton_iterations;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::cube_mapping ( bool enabled )
  {
    _cube_mapping = enabled;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool                  
  surface_renderer::cube_mapping ( ) const
  {
    return _cube_mapping;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::sphere_mapping ( bool enabled )
  {
    _sphere_mapping = enabled;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool                  
  surface_renderer::sphere_mapping ( ) const
  {
    return _sphere_mapping;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::diffuse_mapping ( bool enabled )
  {
    _diffuse_mapping = enabled;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool                  
  surface_renderer::diffuse_mapping ( ) const
  {
    return _diffuse_mapping;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::trimming ( bool enabled )
  {
    _trimming_enabled = enabled;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool                  
  surface_renderer::trimming ( ) const
  {
    return _trimming_enabled;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::trim_approach ( surface_renderer::trimapproach_type a )
  {
    _trimapproach = a;
  }


  /////////////////////////////////////////////////////////////////////////////
  surface_renderer::trimapproach_type         
  surface_renderer::trim_approach ( ) const
  {
    return _trimapproach;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                  
  surface_renderer::raycasting ( bool enabled )
  {
    _raycasting_enabled = enabled;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool                  
  surface_renderer::raycasting ( ) const
  {
    return _raycasting_enabled;
  }


} // namespace gpucast
