/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : stereo_anaglyph.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "gpucast/gl/glut/display/stereo_anaglyph.hpp"

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

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
stereo_anaglyph::stereo_anaglyph( unsigned         width,
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
  rght_eye_texid_ (   ),
  filter_left_    ( stereo_anaglyph::filter_red() ),
  filter_right_   ( stereo_anaglyph::filter_cyan() )
{
  setup();
}

///////////////////////////////////////////////////////////////////////////////
stereo_anaglyph::~stereo_anaglyph()
{}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::setup()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
stereo_anaglyph::display()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */
void
stereo_anaglyph::resize(unsigned width, unsigned height)
{
  displaysetup::resize(width, height);
  initFBO();
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::render()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::render_eye( GLenum colorbuffer, float eye_offset_x)
{
  throw std::runtime_error("not implemented");
}


///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::initFBO()
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::initTexture(GLint texid, int width, int height)
{
  throw std::runtime_error("not implemented");
}

///////////////////////////////////////////////////////////////////////////////
float
stereo_anaglyph::eyedistance( ) const
{
  return eyedistance_;
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::eyedistance( float e)
{
  eyedistance_ = e;
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::filter_left( vec3f const& filter)
{
  filter_left_ = filter;
}

///////////////////////////////////////////////////////////////////////////////
void
stereo_anaglyph::filter_right( vec3f const& filter)
{
  filter_right_ = filter;
}

///////////////////////////////////////////////////////////////////////////////
/* static */  vec3f
stereo_anaglyph::filter_red( )
{
  return vec3f(1.0, 0.0, 0.0);
}

///////////////////////////////////////////////////////////////////////////////
/* static */ vec3f
stereo_anaglyph::filter_cyan( )
{
  return vec3f(0.0, 1.0, 1.0);
}

} } // namespace gpucast / namespace gl
