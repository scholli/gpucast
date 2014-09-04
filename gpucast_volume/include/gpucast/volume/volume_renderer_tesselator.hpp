/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_tesselator.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_TESSELATOR_HPP
#define GPUCAST_VOLUME_RENDERER_TESSELATOR_HPP

// header, system

// header, external
#include <memory>
#include <boost/unordered_map.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/texture1d.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/volume_renderer.hpp>

namespace gpucast 
{

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME volume_renderer_tesselator : public volume_renderer
{
public : // enums, typedefs

  /////////////////////////////////////////////////////////////////////////////
  // per drawble ressource
  /////////////////////////////////////////////////////////////////////////////
  struct drawable_ressource_impl;

  typedef std::shared_ptr<drawable_ressource_impl>                                drawable_ressource_ptr;
  typedef boost::unordered_map<drawable_ptr, drawable_ressource_ptr>                drawable_map;
  typedef drawable_map::value_type                                                  drawable_ressource_pair;
  typedef gpucast::math::axis_aligned_boundingbox<gpucast::math::point3f>                               bbox4f;

  typedef boost::unordered_map<std::string, std::shared_ptr<gpucast::gl::arraybuffer>>   attributebuffer_map;
  //typedef std::map<std::string, std::shared_ptr<gpucast::gl::arraybuffer>>               attributebuffer_map;

public : // c'tor / d'tor

  volume_renderer_tesselator    ( int argc, char** argv );
  ~volume_renderer_tesselator   ();

public : // methods

  virtual void                            clear                   ();

  virtual void                            draw                    ();

  virtual void                            draw                    ( drawable_ptr const& object );

  /* virtual */ void                      transform               ( drawable_ptr const& object, gpucast::math::matrix4f const& m );

  virtual void                            recompile               ();

  void                                    update_attribute_bounds ();

protected : // auxilliary methods

  virtual void                            _initialize_gl_resources ();

  virtual void                            _init                   ();
  virtual void                            _init_shader            ();

private :  // auxilliary methods

protected : // attributes

  bool                                    _initialized;
  drawable_map                            _drawables;
  
  gpucast::gl::vertexarrayobject                 _vao;
  gpucast::gl::arraybuffer                       _vertices;    // vertices
  attributebuffer_map                     _attributes;  // discretized attributes
  gpucast::gl::elementarraybuffer                _indexarray;

  // parameters 
  unsigned                                _tesselation_depth;
  bool                                    _backface_culling;
  
  std::shared_ptr<gpucast::gl::program>        _render_pass;
};

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_TESSELATOR_HPP
