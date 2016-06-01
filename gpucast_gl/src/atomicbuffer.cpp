/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : atomicbuffer.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#include "gpucast/gl/atomicbuffer.hpp"

// header system

// header dependencies
#include <GL/glew.h>

// header project

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
  atomicbuffer::atomicbuffer()
: buffer  ()
{}


///////////////////////////////////////////////////////////////////////////////
  atomicbuffer::atomicbuffer(std::size_t bytes, GLenum usage)
: buffer ( bytes, usage )
{}


///////////////////////////////////////////////////////////////////////////////
  atomicbuffer::~atomicbuffer()
{
  unbind();
}


///////////////////////////////////////////////////////////////////////////////
void            
atomicbuffer::swap(atomicbuffer& t)
{
  buffer::swap(t);
}


///////////////////////////////////////////////////////////////////////////////
void            
atomicbuffer::bind() const
{
  glBindBuffer (target(), id());
}

///////////////////////////////////////////////////////////////////////////////
void atomicbuffer::bind_buffer_range(unsigned binding_point, unsigned offset, unsigned count) const
{
  glBindBufferRange(target(), binding_point, id(), offset, count);
}

///////////////////////////////////////////////////////////////////////////////
void
atomicbuffer::bind_buffer_base(unsigned binding_point) const
{
  glBindBufferBase(target(), binding_point, id());
}


///////////////////////////////////////////////////////////////////////////////
void            
atomicbuffer::unbind() const
{
  glBindBuffer (target(), 0);
}


///////////////////////////////////////////////////////////////////////////////
GLenum          
atomicbuffer::target() const
{
  return GL_ATOMIC_COUNTER_BUFFER;
}
 
} } // namespace gpucast / namespace gl
