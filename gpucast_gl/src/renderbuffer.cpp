/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : renderbuffer.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// i/f include
#include "gpucast/gl/renderbuffer.hpp"

// system includes
#include <iostream>
#include <cassert>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  renderbuffer::renderbuffer()
    : _id(0)
  {
    glGenRenderbuffersEXT(1, &_id);
  }


  ///////////////////////////////////////////////////////////////////////////////
  renderbuffer::renderbuffer(GLenum internal_format, std::size_t width, std::size_t height)
    : _id(0)
  {
    glGenRenderbuffersEXT(1, &_id);

    set(internal_format, width, height);
  }


  ///////////////////////////////////////////////////////////////////////////////
  renderbuffer::~renderbuffer()
  {
    glDeleteRenderbuffersEXT(1, &_id);
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  renderbuffer::bind() const
  {
    glBindRenderbufferEXT(target(), _id);
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  renderbuffer::unbind() const
  {
    glBindRenderbufferEXT(target(), 0);
  }


  ///////////////////////////////////////////////////////////////////////////////
  bool
  renderbuffer::bound() const
  {
    GLint currently_bound_renderbuffer = 0;
    glGetIntegerv( GL_RENDERBUFFER_BINDING_EXT, &currently_bound_renderbuffer );

    return currently_bound_renderbuffer == int(_id);
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  renderbuffer::set(GLenum internal_format, std::size_t width, std::size_t height) const
  {
    assert (int(width) <= renderbuffer::get_maxsize() && int(height) <= renderbuffer::get_maxsize());

    bind();
    glRenderbufferStorageEXT(target(), internal_format, GLsizei(width), GLsizei(height) );
    unbind();
  }


  ///////////////////////////////////////////////////////////////////////////////
  GLuint renderbuffer::id() const
  {
    return _id;
  }


  ///////////////////////////////////////////////////////////////////////////////
  GLint renderbuffer::get_maxsize()
  {
    GLint max_attach = 0;
    glGetIntegerv( GL_MAX_RENDERBUFFER_SIZE_EXT, &max_attach );
    return max_attach;
  }


  ///////////////////////////////////////////////////////////////////////////////
  GLenum
  renderbuffer::target() const
  {
    return TARGET;
  }

} } // namespace gpucast / namespace gl
