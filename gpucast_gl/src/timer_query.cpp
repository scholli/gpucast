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
void timer_query::begin() {
  _result_fetched = false;
  glBeginQuery(GL_TIME_ELAPSED, id());
}

////////////////////////////////////////////////////////////////////////////////
void timer_query::end() {  
  glEndQuery(GL_TIME_ELAPSED);
}

////////////////////////////////////////////////////////////////////////////////
bool timer_query::result_fetched() const
{
  return _result_fetched;
}

////////////////////////////////////////////////////////////////////////////////
double timer_query::time_in_ms(bool wait, double timeout_ms) 
{
  if (wait) {
    auto start = std::chrono::system_clock::now();

    while (!is_available())
    {
      auto end = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      if (elapsed.count() > timeout_ms) {
        BOOST_LOG_TRIVIAL(info) << "timer_query::time_in_ms(): Time out reached.";
        _result_fetched = true;
        return 0.0;
      } 
    };
  }
  GLuint64 result = 0;
  glGetQueryObjectui64v(id(), GL_QUERY_RESULT, &result);
  _result_fetched = true;
  return double(result) / double(10e6);
}

} } // namespace gpucast / namespace gl
