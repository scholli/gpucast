/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : timer_guard.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/util/timer_guard.hpp"

// header, system
#include <boost/log/trivial.hpp>

namespace gpucast { namespace gl {

  ////////////////////////////////////////////////////////////////////////////////
  timer_guard::timer_guard(std::string const& m)
      : _message(m),
        _timer()
  {
      _timer.start();
  }

  ////////////////////////////////////////////////////////////////////////////////
  timer_guard::~timer_guard() 
  {
    _timer.stop();
    BOOST_LOG_TRIVIAL(info) << _message << _timer.result().as_seconds() * 1000.0 << " ms";
  }
 

} } // namespace gpucast / namespace gl

