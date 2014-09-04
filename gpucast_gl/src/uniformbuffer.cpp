/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : uniformbuffer.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/uniformbuffer.hpp"

// header system
#include <vector>
#include <iostream>
#include <exception>

// header dependencies

// header project
#include <gpucast/math/vec4.hpp>
//#include <gpucast/gl/state.hpp>


namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
uniformbuffer::uniformbuffer()
: buffer()
{}


///////////////////////////////////////////////////////////////////////////////
uniformbuffer::uniformbuffer                  ( std::size_t bytes, GLenum usage )
: buffer   ( bytes, usage )
{}


///////////////////////////////////////////////////////////////////////////////
uniformbuffer::~uniformbuffer                 ( )
{}


///////////////////////////////////////////////////////////////////////////////
void            
uniformbuffer::swap( uniformbuffer& t)
{
  buffer::swap(t);
}


///////////////////////////////////////////////////////////////////////////////
void            
uniformbuffer::bind() const
{
  glBindBuffer (target(), id());
}


///////////////////////////////////////////////////////////////////////////////
void            
uniformbuffer::bindbase(GLuint base) const
{
  glBindBufferBase (target(), base, id());
}


///////////////////////////////////////////////////////////////////////////////
void            
uniformbuffer::unbind() const
{
  glBindBuffer (target(), 0);
}


///////////////////////////////////////////////////////////////////////////////
GLenum 
uniformbuffer::target() const
{
  return GL_UNIFORM_BUFFER_EXT;
}


} } // namespace gpucast / namespace gl
