/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : texture3d.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/texture3d.hpp"

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

#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>

// header project
#include <gpucast/gl/math/vec3.hpp>

#include <FreeImagePlus.h>


namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
texture3d::texture3d( )
: _id     (   ),
  _unit   (-1 ),
  _width  ( 0 ),
  _height ( 0 ),
  _depth  ( 0 )
{
  glGenTextures(1, &_id);
}


///////////////////////////////////////////////////////////////////////////////
texture3d::texture3d( std::string const& filename )
: _id     (   ),
  _unit   (-1 ),
  _width  ( 0 ),
  _height ( 0 ),
  _depth  ( 0 )
{
  glGenTextures(1, &_id);
  load( filename, 0 );
}


///////////////////////////////////////////////////////////////////////////////
texture3d::texture3d ( std::size_t  width,
                       std::size_t  height,
                       std::size_t  depth,
                       GLint        internal_format,
                       GLenum       format,
                       GLenum       value_type,
                       GLint        level,
                       GLboolean    border )
: _id     ( 0 ),
  _unit   ( -1 ),
  _width  (  GLsizei(width) ),
  _height (  GLsizei(height) ),
  _depth  (  GLsizei(depth) )
{
  glGenTextures(1, &_id);
  teximage(level, internal_format, _width, _height, _depth, border, format, value_type, 0);
}



///////////////////////////////////////////////////////////////////////////////
texture3d::~texture3d( )
{
  glDeleteTextures(1, &_id);
}

///////////////////////////////////////////////////////////////////////////////
void
texture3d::swap( texture3d& t)
{
  std::swap(_id    , t._id);
  std::swap(_width , t._width);
  std::swap(_height, t._height);
  std::swap(_unit  , t._unit);
}

///////////////////////////////////////////////////////////////////////////////
bool 
texture3d::load( std::string const& in_image_path, unsigned depth )
{
  std::shared_ptr<fipImage>   in_image(new fipImage);
 
  if ( !in_image->load ( in_image_path.c_str()) ) 
  {
    std::cerr << "texture2d::load(): " << "unable to open file: " << in_image_path << std::endl;
    return false;
  }
 
  FREE_IMAGE_TYPE image_type            = in_image->getImageType();
  unsigned        image_width           = in_image->getWidth();
  unsigned        image_height          = in_image->getHeight();
  GLenum          image_internal_format = 0;
  GLenum          image_format          = 0;
  GLenum          image_value_type      = 0;
   
  switch (image_type)  
  {
    case FIT_BITMAP: 
    {
      unsigned num_components = in_image->getBitsPerPixel() / 8;
      switch (num_components) 
      {
        case 1:         image_internal_format = GL_R8;      image_format = GL_RED;  image_value_type = GL_UNSIGNED_BYTE; break;
        case 2:         image_internal_format = GL_RG8;     image_format = GL_RG;   image_value_type = GL_UNSIGNED_BYTE; break;
        case 3:         image_internal_format = GL_RGB8;    image_format = GL_BGR;  image_value_type = GL_UNSIGNED_BYTE; break;
        case 4:         image_internal_format = GL_RGBA8;   image_format = GL_BGRA; image_value_type = GL_UNSIGNED_BYTE; break;
      }
    } break;
    case FIT_INT16:     image_internal_format = GL_RG16I;   image_format = GL_RG;   image_value_type = GL_UNSIGNED_SHORT;  break;
    case FIT_UINT16:    image_internal_format = GL_R16;     image_format = GL_RED;  image_value_type = GL_UNSIGNED_SHORT;  break;
    case FIT_RGB16:     image_internal_format = GL_RGB16;   image_format = GL_RGB;  image_value_type = GL_UNSIGNED_SHORT;  break;
    case FIT_RGBA16:    image_internal_format = GL_RGBA16;  image_format = GL_RGBA; image_value_type = GL_UNSIGNED_SHORT;  break;
    case FIT_INT32:     image_internal_format = GL_R32I;    image_format = GL_RED;  image_value_type = GL_UNSIGNED_INT;    break;
    case FIT_UINT32:    image_internal_format = GL_R32UI;   image_format = GL_RED;  image_value_type = GL_UNSIGNED_INT;    break;
    case FIT_FLOAT:     image_internal_format = GL_R32F;    image_format = GL_RED;  image_value_type = GL_FLOAT;           break;
    case FIT_RGBF:      image_internal_format = GL_RGB32F;  image_format = GL_RGB;  image_value_type = GL_FLOAT;           break;
    case FIT_RGBAF:     image_internal_format = GL_RGBA32F; image_format = GL_RGBA; image_value_type = GL_FLOAT;           break;
    default : std::cerr << "texture2d::load(): " << " unknown image format in file: " << in_image_path << std::endl; return false;
  }
 
  teximage(0, image_internal_format, image_width, image_height, depth, 0, image_format, image_value_type, static_cast<void*>(in_image->accessPixels()));

  return true;
}

///////////////////////////////////////////////////////////////////////////////
void
texture3d::resize( std::size_t  width,
                   std::size_t  height,
                   std::size_t  depth, 
                   GLint        internal_format)
{
  _width  = GLsizei(width); 
  _height = GLsizei(height); 
  _depth  = GLsizei(depth);

#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glTextureImage3DEXT( _id, target(), 0, internal_format, _width, _height, _depth, 0, GL_RGBA, GL_UNSIGNED_INT, 0);
#else
  bind();
  glTexImage3D(target(), 0, internal_format, GLsizei(width), GLsizei(height), GLsizei(depth), 0, GL_RGBA, GL_FLOAT, 0);
  unbind();
#endif
}


///////////////////////////////////////////////////////////////////////////////
void
texture3d::teximage( GLint    level,
                     GLint    internal_format,
			               GLsizei  width,
			               GLsizei  height,
                     GLsizei  depth, 
			               GLint    border,
			               GLenum   format,
			               GLenum   type,
			               GLvoid*  pixels )
{
  _width = width; 
  _height = height; 
  _depth = depth;
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glTextureImage3DEXT( _id, target(), 0, internal_format, _width, _height, _depth, 0, format, type, pixels);
#else
  bind();
  glTexImage3D  ( target(), level, internal_format, width, height, depth, border, format, type, pixels );
  unbind();
#endif
}

///////////////////////////////////////////////////////////////////////////////
void
texture3d::bind ( )
{
  glBindTexture(target(), _id);
}

///////////////////////////////////////////////////////////////////////////////
void
texture3d::bind ( GLint texunit )
{
  if ( _unit >= 0 ) {
    unbind();
  }
  _unit = texunit;

#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glBindMultiTextureEXT(GL_TEXTURE0 + _unit, target(), _id);
  //glEnableIndexedEXT(target(), _unit);
#else
  glActiveTexture(GL_TEXTURE0 + _unit);
  glEnable(target());
  glBindTexture(target(), _id);
#endif
}

///////////////////////////////////////////////////////////////////////////////
void
texture3d::unbind ()
{
  if (_unit >= 0)
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glBindMultiTextureEXT(GL_TEXTURE0 + _unit, target(), 0);
    //glDisableIndexedEXT(target(), _unit);
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
texture3d::set_parameteri( GLenum pname, GLint value )
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
texture3d::set_parameterf( GLenum pname, GLfloat value )
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
texture3d::set_parameterfv ( GLenum pname, GLfloat const* value )
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
texture3d::id() const
{
  return _id;
}
///////////////////////////////////////////////////////////////////////////////
/* static */ GLenum
texture3d::target ( )
{
  return GL_TEXTURE_3D;
}

} } // namespace gpucast / namespace gl
