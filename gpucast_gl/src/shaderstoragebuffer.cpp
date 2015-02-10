/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : arraybuffer.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/shaderstoragebuffer.hpp"

// header system
#include <vector>
#include <iostream>
#include <exception>

// header dependencies
#include <GL/glew.h>

// header project
#include <gpucast/math/vec4.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
  shaderstoragebuffer::shaderstoragebuffer()
: buffer  ()
{}


///////////////////////////////////////////////////////////////////////////////
  shaderstoragebuffer::shaderstoragebuffer(std::size_t bytes, GLenum usage)
: buffer ( bytes, usage )
{}


///////////////////////////////////////////////////////////////////////////////
  shaderstoragebuffer::~shaderstoragebuffer()
{
  unbind();
}


///////////////////////////////////////////////////////////////////////////////
void            
shaderstoragebuffer::swap(shaderstoragebuffer& t)
{
  buffer::swap(t);
}


///////////////////////////////////////////////////////////////////////////////
void            
shaderstoragebuffer::bind() const
{
  glBindBuffer (target(), id());
}

///////////////////////////////////////////////////////////////////////////////
void 
shaderstoragebuffer::bind_buffer_base(unsigned binding_point) const
{
  glBindBufferBase(target(), binding_point, id());
}

///////////////////////////////////////////////////////////////////////////////
void            
shaderstoragebuffer::unbind() const
{
  glBindBuffer (target(), 0);
}


///////////////////////////////////////////////////////////////////////////////
GLenum          
shaderstoragebuffer::target() const
{
  return GL_SHADER_STORAGE_BUFFER;
}
 
} } // namespace gpucast / namespace gl
