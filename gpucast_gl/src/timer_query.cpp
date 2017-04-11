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
#include <chrono>
#include <boost/log/trivial.hpp>
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
double timer_query::time_in_ms(bool wait, double timeout_ms) const
{
  if (wait) {
    auto start = std::chrono::system_clock::now();

    while (!is_available())
    {
      auto end = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      if (elapsed.count() > timeout_ms) {
        BOOST_LOG_TRIVIAL(info) << "timer_query::time_in_ms(): Time out reached.";
        return 0.0;
      } 
    };
  }
  GLuint64 result = 0;
  glGetQueryObjectui64v(id(), GL_QUERY_RESULT, &result);
  return double(result) / double(10e6);
}

} } // namespace gpucast / namespace gl
