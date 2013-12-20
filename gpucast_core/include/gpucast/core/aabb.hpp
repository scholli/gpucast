/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : aabb.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_AABB_HPP
#define GPUCAST_CORE_AABB_HPP

// header, system

// header, exernal
#include <memory>

#include <gpucast/math/oriented_boundingbox.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/pointmesh3d.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/math/vec4.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

namespace gpucast {

class GPUCAST_CORE aabb : public gpucast::math::axis_aligned_boundingbox<gpucast::math::point<float, 3> >
{
  public : // typedefs / enums

    typedef gpucast::math::axis_aligned_boundingbox<gpucast::math::point<float, 3> >  base_type;

  public : // c'tor / d'tor

    aabb   ( );
    aabb   ( base_type const& );
    ~aabb  ();

  public : // methods

    void                      draw  ( gpucast::gl::matrix4x4<float> const& mvp );
    void                      color ( gpucast::gl::vec4f const& color );

  private : // auxilliary methods

    void                      _init ();

  private : // attributes

    bool                                              _initialized;

    std::shared_ptr<gpucast::gl::arraybuffer>         _vertexarray;
    std::shared_ptr<gpucast::gl::arraybuffer>         _colorarray;
    std::shared_ptr<gpucast::gl::vertexarrayobject>   _arrayobject;

    std::shared_ptr<gpucast::gl::program>             _program; 
    gpucast::gl::vec4f                                _color;

};

} // namespace gpucast

#endif // GPUCAST_CORE_AABB_HPP