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
#ifndef GPUCAST_OBB_HPP
#define GPUCAST_OBB_HPP

// header, system

// header, exernal
#include <memory>

#include <gpucast/math/oriented_boundingbox.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/pointmesh3d.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/math/vec4.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/beziervolume.hpp>



namespace gpucast {

class GPUCAST_VOLUME obb : public gpucast::math::oriented_boundingbox<gpucast::math::point<beziervolume::value_type, 3> >
{
  public : // typedefs / enums

    typedef gpucast::math::oriented_boundingbox<gpucast::math::point<beziervolume::value_type, 3> > base_type;

  public : // c'tor / d'tor

    obb   ( );

    obb   ( base_type const& );

    template <template <typename T> class build_policy>
    obb   ( gpucast::math::pointmesh3d<gpucast::math::point3d> const& mesh, build_policy<gpucast::math::point3d> policy );

    template <typename iterator_t, template <typename T> class build_policy>
    obb   ( iterator_t begin, iterator_t end, build_policy<gpucast::math::point3d> policy );

    ~obb  ();

  public : // methods

    void                      draw  ( gpucast::math::matrix4x4<float> const& mvp );
    void                      color ( gpucast::math::vec4f const& color );

  private : // auxilliary methods

    void                      _init ();

  private : // attributes

    bool                      _initialized;

    std::shared_ptr<gpucast::gl::arraybuffer>         _vertexarray;
    std::shared_ptr<gpucast::gl::arraybuffer>         _colorarray;
    std::shared_ptr<gpucast::gl::vertexarrayobject>   _arrayobject;

    std::shared_ptr<gpucast::gl::program>             _program;             
    gpucast::math::vec4f                                  _color;

};


////////////////////////////////////////////////////////////////////////////////
template <template <typename T> class build_policy>
obb::obb ( gpucast::math::pointmesh3d<gpucast::math::point3d> const& mesh, build_policy<gpucast::math::point3d> policy )
  : base_type     ( mesh, policy ),
    _initialized  ( false ),
    _vertexarray  (),
    _colorarray   (),
    _arrayobject  (),
    _program      ()
{}


////////////////////////////////////////////////////////////////////////////////
template <typename iterator_t, template <typename T> class build_policy>
obb::obb ( iterator_t begin, iterator_t end, build_policy<gpucast::math::point3d> policy )
  : base_type     ( begin, end, policy ),
    _initialized  ( false ),
    _vertexarray  (),
    _colorarray   (),
    _arrayobject  (),
    _program      ()
{}


} // namespace gpucast

#endif // GPUCAST_OBB_HPP