/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : query.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/query.hpp"

// header system
#include <GL/glew.h>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
query::query()
: _id(0U)
{
  glGenQueries(1, &_id);
}

////////////////////////////////////////////////////////////////////////////////
query::~query()
{
  glDeleteQueries(1, &_id);
}

////////////////////////////////////////////////////////////////////////////////
bool query::is_available() const {
  GLuint64 result = GL_FALSE;
  glGetQueryObjectui64v(id(), GL_QUERY_RESULT_AVAILABLE, &result);
  return result != GL_FALSE;
}

////////////////////////////////////////////////////////////////////////////////
void query::reset()
{
  glDeleteQueries(1, &_id);
  glGenQueries(1, &_id);
}

////////////////////////////////////////////////////////////////////////////////
unsigned query::id() const {
  return _id;
}

} } // namespace gpucast / namespace gl
