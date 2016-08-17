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
double timer_query::time_in_ms(bool wait) const 
{
  if (wait) {
    while (!is_available());
  }
  GLuint64 result = 0;
  glGetQueryObjectui64v(id(), GL_QUERY_RESULT, &result);
  return double(result) / double(10e6);
}

} } // namespace gpucast / namespace gl
