/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : surface_renderer_gl.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_RENDERSERVICE_GLPP_HPP
#define GPUCAST_CORE_RENDERSERVICE_GLPP_HPP

// header, system
#include <unordered_map>

// header, external
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/cubemap.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/sampler.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/surface_renderer.hpp>



namespace gpucast 
{


///////////////////////////////////////////////////////////////////////////////
class GPUCAST_CORE surface_renderer_gl : public surface_renderer
{
public : // enums, typedefs

  struct drawable_ressource_impl;

  typedef std::shared_ptr<drawable_ressource_impl>                    drawable_ressource_ptr;
  typedef std::unordered_map<drawable_ptr, drawable_ressource_ptr>    drawable_map;
  typedef drawable_map::value_type                                      drawable_ressource_pair;

public : // c'tor / d'tor

  surface_renderer_gl    ( int argc, char** argv );

  ~surface_renderer_gl   ();

public : // methods

  /* virtual */ drawable_ptr    create      ();

  void                          clear       ();

  /* virtual */ void            draw        ();

  // /* virtual */ void            draw        ( drawable_ptr const& drawable );

  void                          memory_usage( std::size_t& trim_data_binarypartition_bytes,
                                              std::size_t& trim_data_contourmap_bytes,
                                              std::size_t& surface_data_bytes ) const;

  /* virtual */ void            spheremap   ( std::string const& filepath );

  /* virtual */ void            diffusemap  ( std::string const& filepath );

  /* virtual */ void            cubemap     ( std::string const& positive_x,
                                              std::string const& negative_x,
                                              std::string const& positive_y,
                                              std::string const& negative_y,
                                              std::string const& positive_z,
                                              std::string const& negative_z );

  void                          recompile   ();

  virtual void                  _init_shader  ();

private : // auxilliary methods

  void                          _sync         ( drawable_ressource_pair const& );

private : // attributes

  drawable_map                        _drawables;

  // surface_renderer global ressources
  std::shared_ptr<gpucast::gl::program>    _program;
  std::shared_ptr<gpucast::gl::cubemap>    _cubemap;
  std::shared_ptr<gpucast::gl::texture2d>  _spheremap;
  std::shared_ptr<gpucast::gl::texture2d>  _diffusemap;
  std::shared_ptr<gpucast::gl::sampler>    _linear_interp;
};

} // namespace gpucast

#endif // GPUCAST_CORE_RENDERSERVICE_GLPP_HPP
