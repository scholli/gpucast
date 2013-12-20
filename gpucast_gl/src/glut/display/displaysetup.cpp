/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : displaysetup.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "gpucast/gl/glut/display/displaysetup.hpp"

#include <iostream>

// header, system
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

// header, project
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/camera.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
displaysetup::displaysetup(std::string const&   name,
                           unsigned             width,
                           unsigned             height,
                           float                scrwidth,
                           float                scrheight,
                           vec3f const&         camera_position,
                           vec3f const&         screen_position )
  : _name           ( name ),
    _width          ( width ),
    _height         ( height ),
    _draw_fun       ( ),
    _userdata       ( 0 ),
    _screenwidth    ( scrwidth ),
    _screenheight   ( scrheight ),
    _camera         ( 0 ),
    _trackball      ( new trackball )
{
  _camera->target        ( screen_position );
  _camera->position      ( camera_position );
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ displaysetup::~displaysetup()
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
displaysetup::resize(unsigned width, unsigned height)
{
  _width  = width;
  _height = height;
}

///////////////////////////////////////////////////////////////////////////////
unsigned
displaysetup::width() const
{
  return _width;
}

///////////////////////////////////////////////////////////////////////////////
unsigned
displaysetup::height() const
{
  return _height;
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup::viewsetup(float px, float py, float pz, float targetx, float targety, float targetz)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // distance to screen
  float const d = (vec3f(px, py, pz) - vec3f(targetx, targety, targetz)).length();

  glFrustum( ( _camera->screenoffset_x() - _screenwidth/2  ) * _camera->nearplane() / d,
             ( _camera->screenoffset_x() + _screenwidth/2  ) * _camera->nearplane() / d,
             ( _camera->screenoffset_y() - _screenheight/2 ) * _camera->nearplane() / d,
             ( _camera->screenoffset_y() + _screenheight/2 ) * _camera->nearplane() / d,
               _camera->nearplane(),
               _camera->farplane () );

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gluLookAt( px             , py             , pz,
             targetx        , targety        , targetz,
             _camera->up()[0], _camera->up()[1], _camera->up()[2]);
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup::set_drawfun( std::function<void (void*)> draw_fun, void* userdata )
{
  _draw_fun = draw_fun;
  _userdata = userdata;
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup::set_camera( camera* cam )
{
  _camera = cam;
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup::mouse ( int button, int state, int x ,int y) const
{
  _trackball->mouse(eventhandler::button(button), eventhandler::state(state), x, y);
}

///////////////////////////////////////////////////////////////////////////////
void
displaysetup::motion (int x ,int y) const
{
  _trackball->motion(x, y);
}

} } // namespace gpucast / namespace gl
