/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : singleview.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "gpucast/gl/glut/display/singleview.hpp"

// header, system
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

// header, project
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/camera.hpp>

namespace gpucast { namespace gl {


///////////////////////////////////////////////////////////////////////////////
singleview::singleview( unsigned      width,
                        unsigned      height,
                        vec3f const&  camera_position,
                        vec3f const&  screen_position,
                        float         screenwidth,
                        float         screenheight )
  : displaysetup("Singleframe Setup", width, height, screenwidth, screenheight, camera_position, screen_position)
{
  resize(width, height);
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */
singleview::~singleview()
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
singleview::display()
{
  // clear buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  matrix4x4<float> const rot = _trackball->rotation();
  glMultMatrixf(&rot[0]);

  viewsetup( _camera->position()[0],
             _camera->position()[1],
             _camera->position()[2],
             _camera->target()[0],
             _camera->target()[1],
             _camera->target()[2]);

  // save the initial ModelView matrix before modifying ModelView matrix
  glMatrixMode(GL_MODELVIEW);

  glPushMatrix();
  {
    // transform camera
    glTranslatef(_trackball->shiftx(), _trackball->shifty(), _trackball->distance());

    matrix4x4<float> const rot = _trackball->rotation();
    glMultMatrixf(&rot[0]);

    _draw_fun(_userdata);
  }
  glPopMatrix();
  glutSwapBuffers();
}

} } // namespace gpucast / namespace gl
