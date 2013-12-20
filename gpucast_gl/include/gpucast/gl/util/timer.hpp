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
#ifndef GPUCAST_GL_TIMER_HPP
#define GPUCAST_GL_TIMER_HPP

// header, system
#include <utility>
#include <iostream>

// header, project
#ifndef WIN32
  #include <sys/time.h>
#else
  #include <WinSock.h>
#endif

// header, project
#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

    class GPUCAST_GL time_duration
    {
    public:
      typedef long long LONGLONG;

      time_duration(LONGLONG h = 0, LONGLONG m = 0, LONGLONG s = 0, double fractional = 0.0);

      double as_seconds() const;

      LONGLONG hours;
      LONGLONG minutes;
      LONGLONG seconds;
      double   fractional_seconds;
    };

    GPUCAST_GL time_duration operator-(time_duration const& lhs, time_duration const& rhs);
    GPUCAST_GL time_duration operator+(time_duration const& lhs, time_duration const& rhs);
    GPUCAST_GL time_duration operator/(time_duration const& lhs, double rhs);
    GPUCAST_GL std::ostream& operator<<(std::ostream& os, time_duration const& t);


    class GPUCAST_GL timer
    {
    public:

      timer();
      ~timer();

    public:

      void             frame(std::ostream& os = std::cout, double integration_time_in_s = 1.0);
      void             time(time_duration& time_duration);

      void             start();
      void             stop();
      time_duration    result() const;

    private: // methods



    private: // members

      unsigned int   _framecount;
      time_duration  _last_frame_integration;

      time_duration  _starttime;
      time_duration  _stoptime;
    };

} } // namespace gpucast / namespace gl

#endif // TIMER_HPP

