/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : camera.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/util/camera.hpp"

#include <gpucast/gl/glpp.hpp>
#include <GL/freeglut.h>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  camera::camera()
    : _target        ( 0.0, 0.0,-1.0 ),
      _position      ( 0.0, 0.0, 0.0 ),
      _up            ( 0.0, 1.0, 0.0 ),
      _znear         ( 10.0 ),
      _zfar          ( 2000.0 ),
      _screenoffset_y( 0.0 ),
      _screenoffset_x( 0.0 ),
      _screenwidth   ( 40.0 ),
      _screenheight  ( 30.0 ),
      _screendistance( 50.0 ),
      _resolution_x  ( 512 ),
      _resolution_y  ( 512 )
  {}


  ///////////////////////////////////////////////////////////////////////////////
  camera::~camera()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  camera::resize( std::size_t resx, std::size_t resy )
  {
    _resolution_x = resx;
    _resolution_y = resy;
  }


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  camera::draw()
  {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glFrustum( ( _screenoffset_x - _screenwidth/2  ) * _znear / _screendistance,
               ( _screenoffset_x + _screenwidth/2  ) * _znear / _screendistance,
               ( _screenoffset_y - _screenheight/2 ) * _znear / _screendistance,
               ( _screenoffset_y + _screenheight/2 ) * _znear / _screendistance,
               _znear,
               _zfar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt( _position[0],  _position[1], _position[2],
               _target[0],    _target[1],   _target[2],
               _up[0],        _up[1],       _up[2]);

    if (_drawcallback)
    {
      _drawcallback();
    }

    glutSwapBuffers();
  }


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  camera::drawcallback ( std::function<void()> drawfun )
  {
    _drawcallback = drawfun;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::target(vec3f const& t)
  {
    _target = t;
  }


  ///////////////////////////////////////////////////////////////////////////////
  vec3f const&
  camera::target() const
  {
    return _target;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::position(vec3f const& pos)
  {
    _position = pos;
  }


  ///////////////////////////////////////////////////////////////////////////////
  vec3f const&
  camera::position() const
  {
    return _position;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::up(vec3f const& upvec)
  {
    _up = upvec;
  }


  ///////////////////////////////////////////////////////////////////////////////
  vec3f const&
  camera::up( ) const
  {
    return _up;
  }

  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::nearplane(float n)
  {
    _znear = n;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::nearplane() const
  {
    return _znear;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::farplane(float f)
  {
    _zfar = f;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::farplane() const
  {
    return _zfar;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::screenoffset_y(float oy)
  {
    _screenoffset_y = oy;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::screenoffset_y() const
  {
    return _screenoffset_y;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::screenoffset_x(float ox)
  {
    _screenoffset_x = ox;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::screenoffset_x() const
  {
    return _screenoffset_x;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::screenwidth ( float w )
  {
    _screenwidth = w;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::screenwidth ( ) const
  {
    return _screenwidth;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::screenheight ( float h )
  {
    _screenheight = h;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::screenheight ( ) const
  {
    return _screenheight;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  camera::screendistance ( float d )
  {
    _screendistance = d;
  }


  ///////////////////////////////////////////////////////////////////////////////
  float
  camera::screendistance ( ) const
  {
    return _screendistance;
  }


  ///////////////////////////////////////////////////////////////////////////////
  std::size_t
  camera::resolution_x() const
  {
    return _resolution_x;
  }


  ///////////////////////////////////////////////////////////////////////////////
  std::size_t
  camera::resolution_y() const
  {
    return _resolution_y;
  }

} } // namespace gpucast / namespace gl

