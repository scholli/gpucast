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
#include "gpucast/gl/arraybuffer.hpp"

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
arraybuffer::arraybuffer()
: buffer  ()
{}


///////////////////////////////////////////////////////////////////////////////
arraybuffer::arraybuffer( std::size_t bytes, GLenum usage )
: buffer ( bytes, usage )
{}


///////////////////////////////////////////////////////////////////////////////
arraybuffer::~arraybuffer( )
{
  unbind();
}


///////////////////////////////////////////////////////////////////////////////
void            
arraybuffer::swap( arraybuffer& t)
{
  buffer::swap(t);
}


///////////////////////////////////////////////////////////////////////////////
void            
arraybuffer::bind() const
{
  glBindBuffer (target(), id());
}


///////////////////////////////////////////////////////////////////////////////
void            
arraybuffer::unbind() const
{
  glBindBuffer (target(), 0);
}


///////////////////////////////////////////////////////////////////////////////
GLenum          
arraybuffer::target( ) const
{
  return GL_ARRAY_BUFFER;
}
 
} } // namespace gpucast / namespace gl
