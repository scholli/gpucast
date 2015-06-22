/********************************************************************************
*
* Copyright (C) 2015 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : timer_query.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/timer_query.hpp"
#include "gpucast/gl/error.hpp"

// header system
#include <GL/glew.h>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
void timer_query::begin() const {
  glBeginQuery(GL_TIME_ELAPSED, id());
}

////////////////////////////////////////////////////////////////////////////////
void timer_query::end() const {  
  glEndQuery(GL_TIME_ELAPSED);
}

////////////////////////////////////////////////////////////////////////////////
bool timer_query::is_available() const {
  GLuint64 result = GL_FALSE;
  glGetQueryObjectui64v(id(), GL_QUERY_RESULT_AVAILABLE, &result);
  return result != GL_FALSE;
}

////////////////////////////////////////////////////////////////////////////////
double timer_query::result_no_wait() const {
  GLuint64 result = 0;
  glGetQueryObjectui64v(id(), GL_QUERY_RESULT, &result);
  return double(result) / double(10e6);
}

////////////////////////////////////////////////////////////////////////////////
double timer_query::result_wait() const {
  while (!is_available());
  return result_no_wait();
}

} } // namespace gpucast / namespace gl
