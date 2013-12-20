/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : stereo_karo.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "gpucast/gl/glut/display/stereo_karo.hpp"

// header, system
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

// header, project
#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/error.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
stereo_karo::stereo_karo( unsigned         width,
                          unsigned         height,
                          vec3f const&     screenpos,
                          vec3f const&     viewpos,
                          float            screenwidth,
                          float            screenheight)
: displaysetup    ( "Samsung 3DTV", width, height, screenwidth, screenheight, screenpos, viewpos ),
  eyedistance_    ( 60.0 ),
  fbo_            (   ),
  rb_             (   ),
  sb_             (   ),
  program_        ( 0 ),
  left_eye_texid_ (   ),
  rght_eye_texid_ (   )
{
  setup();
}

///////////////////////////////////////////////////////////////////////////////
stereo_karo::~stereo_karo()
{}

///////////////////////////////////////////////////////////////////////////////
void
stereo_karo::setup()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
stereo_karo::display()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */
void
stereo_karo::resize(unsigned width, unsigned height)
{
  displaysetup::resize(width, height);
  initFBO();
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_karo::render()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_karo::render_eye( GLenum colorbuffer, float eye_offset_x)
{
  throw std::runtime_error("not implemented");
}


///////////////////////////////////////////////////////////////////////////////
void
stereo_karo::initFBO()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_karo::initTexture(GLint texid, int width, int height)
{
  throw std::runtime_error("not implemented");
}

} } // namespace gpucast / namespace gl
