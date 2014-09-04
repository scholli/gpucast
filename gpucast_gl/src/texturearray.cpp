/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : texturearray.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/texturearray.hpp"

// header system
#include <fstream>
#include <vector>
#include <iostream>
#include <exception>

// disable boost warnings
#if WIN32
  #pragma warning (disable : 4512)
  #pragma warning (disable : 4127)
#endif

// header dependencies
#include <GL/glew.h>

// header project
#include <gpucast/math/vec3.hpp>


namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
texturearray::texturearray( )
: _id     (   ),
  _unit   ( -1),
  _width  ( 0 ),
  _height ( 0 ),
  _depth  ( 0 )
{
  glGenTextures(1, &_id);
}


///////////////////////////////////////////////////////////////////////////////
texturearray::texturearray( std::size_t width, std::size_t height, std::size_t depth )
: _id     ( 0 ),
  _unit   ( -1 ),
  _width  ( width ),
  _height ( height ),
  _depth  ( depth )
{
  glGenTextures(1, &_id);

  bind();
  glTexImage3D(target(), 0, GL_RGBA8, GLsizei( _width), GLsizei(_height), GLsizei(_depth), 0, GL_RGB, GL_FLOAT, 0);
  unbind();
}


///////////////////////////////////////////////////////////////////////////////
texturearray::texturearray ( std::size_t width,
                             std::size_t height,
                             std::size_t depth,
                             int internal_format,
                             int format,
                             int value_type,
                             int level,
                             bool border )
: _id     ( 0 ),
  _unit   ( -1 ),
  _width  ( width ),
  _height ( height ),
  _depth  ( depth )
{
  glGenTextures(1, &_id);

  bind();
  glTexImage3D(target(), level, internal_format, GLsizei(_width), GLsizei(_height), GLsizei(_depth), border, format, value_type, 0);
  unbind();
}



///////////////////////////////////////////////////////////////////////////////
texturearray::~texturearray( )
{
  glDeleteTextures(1, &_id);
}

///////////////////////////////////////////////////////////////////////////////
void
texturearray::swap( texturearray& t)
{
  std::swap(_id    , t._id);
  std::swap(_width , t._width);
  std::swap(_height, t._height);
  std::swap(_depth,  t._depth);
  std::swap(_unit  , t._unit);
}


///////////////////////////////////////////////////////////////////////////////
void
texturearray::teximage(GLint    level,
                       GLint    internal_format,
			                 GLsizei  width,
			                 GLsizei  height,
                       GLsizei  depth,
			                 GLint    border,
			                 GLenum   format,
			                 GLenum   type,
			                 GLvoid*  pixels )
{
  bind();
  glTexImage3D  ( target(), level, internal_format, width, height, depth, border, format, type, pixels );
  unbind();
}


///////////////////////////////////////////////////////////////////////////////
void
texturearray::texsubimage( GLint    level,
                           GLint  	xoffset, 
 	                         GLint  	yoffset, 
 	                         GLint  	zoffset, 
			                     GLsizei  width,
			                     GLsizei  height,
                           GLsizei  depth,
			                     GLenum   format,
			                     GLenum   type,
			                     GLvoid*  data )
{
  bind();
  glTexSubImage3D  ( target(), level,	xoffset, yoffset, zoffset, width, height, depth, format, type, data );
  unbind();
}


///////////////////////////////////////////////////////////////////////////////
void
texturearray::bind ( )
{
  glBindTexture(target(), _id);
}

///////////////////////////////////////////////////////////////////////////////
void
texturearray::bind ( GLint texunit )
{
  if ( _unit >= 0 ) {
    unbind();
  }
  _unit = texunit;

#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glBindMultiTextureEXT(GL_TEXTURE0 + _unit, target(), _id);
  glEnableIndexedEXT(target(), _unit);
#else
  glActiveTexture(GL_TEXTURE0 + _unit);
  glEnable(target());
  glBindTexture(target(), _id);
#endif
}

///////////////////////////////////////////////////////////////////////////////
void
texturearray::unbind ()
{
  if (_unit >= 0)
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glBindMultiTextureEXT(GL_TEXTURE0 + _unit, target(), 0);
    glDisableIndexedEXT(target(), _unit);
#else
    glActiveTexture(GL_TEXTURE0 + _unit);
    glBindTexture(target(), 0);
    glDisable(target());
#endif
  }
  _unit = -1;
}

///////////////////////////////////////////////////////////////////////////////
void
texturearray::set_parameteri( GLenum pname, GLint value )
{
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glTextureParameteriEXT(_id, target(), pname, value);
#else
  bind();
  glTexParameteri(target(), pname, value);
  unbind();
#endif
}

///////////////////////////////////////////////////////////////////////////////
void
texturearray::set_parameterf( GLenum pname, GLfloat value )
{
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glTextureParameterfEXT(_id, target(), pname, value);
#else
  bind();
  glTexParameterf(target(), pname, value);
  unbind();
#endif
}

///////////////////////////////////////////////////////////////////////////////
void
texturearray::set_parameterfv ( GLenum pname, GLfloat const* value )
{
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glTextureParameterfvEXT(_id, target(), pname, value);
#else
  bind();
  glTexParameterfv(target(), pname, value);
  unbind();
#endif
}

///////////////////////////////////////////////////////////////////////////////
GLuint const
texturearray::id() const
{
  return _id;
}
///////////////////////////////////////////////////////////////////////////////
/* static */ GLenum
texturearray::target ( )
{
  return GL_TEXTURE_2D_ARRAY_EXT;
}

} } // namespace gpucast / namespace gl
