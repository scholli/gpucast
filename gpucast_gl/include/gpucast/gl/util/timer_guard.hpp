/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : timer.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TIMER_GUARD_HPP
#define GPUCAST_GL_TIMER_GUARD_HPP

// header, system

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/timer.hpp>


namespace gpucast { namespace gl {

    class GPUCAST_GL timer_guard
    {
    public:

      timer_guard(std::string const& message);
      ~timer_guard();

    public:

    private: // methods

    private: // members

      std::string _message;
      timer       _timer;
    };

} } // namespace gpucast / namespace gl

#endif // TIMER_GUARD_HPP

