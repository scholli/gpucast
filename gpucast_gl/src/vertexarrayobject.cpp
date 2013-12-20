/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vertexarrayobject.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/vertexarrayobject.hpp"

// header system
#include <iostream>

// header dependencies

// header project
#include <gpucast/gl/buffer.hpp>
#include <gpucast/gl/error.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
vertexarrayobject::vertexarrayobject()
: _id     ()
{ 
  glGenVertexArrays (1, &_id);
}

///////////////////////////////////////////////////////////////////////////////
vertexarrayobject::~vertexarrayobject()
{
  glDeleteVertexArrays (1, &_id);
}


///////////////////////////////////////////////////////////////////////////////
void            
vertexarrayobject::bind() const
{
  glBindVertexArray(_id);
}


///////////////////////////////////////////////////////////////////////////////
void            
vertexarrayobject::unbind() const
{
  glBindVertexArray (0);
}


///////////////////////////////////////////////////////////////////////////////
void
vertexarrayobject::enable_attrib (std::size_t index) const
{
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glEnableVertexArrayAttribEXT(_id, GLuint(index));
#else  
  bind();
  glEnableVertexAttribArray( GLuint(index) );
  unbind();
#endif
}


///////////////////////////////////////////////////////////////////////////////
void
vertexarrayobject::disable_attrib (std::size_t index) const
{
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  glDisableVertexArrayAttribEXT(_id, GLuint(index));
#else  
  bind();
  glDisableVertexAttribArray( GLuint(index) );
  unbind();
#endif
}


///////////////////////////////////////////////////////////////////////////////
void 
vertexarrayobject::attrib_array ( buffer const& buf, std::size_t index, std::size_t size, GLenum type, bool normalized, std::size_t stride, std::size_t offset ) const
{
#if GPUCAST_GL_DIRECT_STATE_ACCESS
  // workaround
  buf.bind();
  switch ( type ) 
  {
    case GL_FLOAT         : glVertexArrayVertexAttribOffsetEXT (GLuint(_id), GLuint(buf.id()), GLuint(index), GLint(size), type, normalized, GLint(stride), GLintptr(offset)); break;
    case GL_BYTE          : glVertexArrayVertexAttribOffsetEXT (GLuint(_id), GLuint(buf.id()), GLuint(index), GLint(size), type, normalized, GLint(stride), GLintptr(offset)); break;
    case GL_UNSIGNED_INT  : glVertexArrayVertexAttribIOffsetEXT(GLuint(_id), GLuint(buf.id()), GLuint(index), GLint(size), type, GLint(stride), GLintptr(offset)); break;
    default : std::cerr << "vertexarrayobject::attrib_array() : type not handled yet" << std::endl;
  }
  buf.unbind();
#else 
  bind();
  buf.bind();
  switch ( type ) 
  {
    case GL_FLOAT         : glVertexAttribPointer(index, size, type, normalized, stride, pointer); break;
    case GL_UNSIGNED_INT  : glVertexAttribIPointer(index, size, type, stride, pointer); break;
    default : std::cerr << "vertexarrayobject::attrib_array() : type not handled yet" << std::endl;
  }
  buf.unbind();
  unbind();
#endif
}


} } // namespace gpucast / namespace gl
