/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : obb.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_OBB_HPP
#define GPUCAST_GL_OBB_HPP

// header, system

// header, exernal
#include <memory>

#include <gpucast/math/oriented_boundingbox.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/math/vec4.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

namespace gpucast {
  namespace gl {

class GPUCAST_GL obb : public gpucast::math::oriented_boundingbox<gpucast::math::point<float, 3> >
{
  public : // typedefs / enums

    typedef gpucast::math::oriented_boundingbox<gpucast::math::point<float, 3> >  base_type;

  public : // c'tor / d'tor

    obb   ( );
    obb( base_type const& );

  public : // methods

    void                      draw  ( gpucast::math::matrix4x4<float> const& mvp, bool wireframe );
    void                      color ( gpucast::math::vec4f const& color );

  private : // auxilliary methods

    void                      _init();
    void                      _init_geometry  ();
    void                      _init_program   ();
    void                      _init_color     ();

  private : // attributes

    gpucast::math::vec4f                       _color;
    std::shared_ptr<gpucast::gl::program>      _program;
    std::shared_ptr<gpucast::gl::cube>         _cube;
};

  } // namespace gl
} // namespace gpucast

#endif // GPUCAST_GL_AABB_HPP