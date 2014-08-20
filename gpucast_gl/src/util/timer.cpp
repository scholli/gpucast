/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : timer.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/util/timer.hpp"

// header, system
#include <cassert>  // assert
#include <iostream> // output
#include <cmath> 
#include <cstdint> 

namespace gpucast { namespace gl {

  ////////////////////////////////////////////////////////////////////////////////
  time_duration::time_duration(LONGLONG h, LONGLONG m, LONGLONG s, double f)
      : hours               ( h ),
        minutes             ( m ),
        seconds             ( s ),
        fractional_seconds  ( f )
    {}

  ////////////////////////////////////////////////////////////////////////////////
  double 
  time_duration::as_seconds() const
  {
    return double(hours) * 3600.0 + double(minutes) * 60.0 + double(seconds) + fractional_seconds;
  }

  ////////////////////////////////////////////////////////////////////////////////
  time_duration operator-(time_duration const& lhs, time_duration const& rhs)
  {
    time_duration t;

    t.fractional_seconds  = lhs.fractional_seconds  - rhs.fractional_seconds;
    t.seconds             = lhs.seconds             - rhs.seconds;
    t.minutes             = lhs.minutes             - rhs.minutes;
    t.hours               = lhs.hours               - rhs.hours;

    if (lhs.fractional_seconds < rhs.fractional_seconds) {
      t.fractional_seconds += 1.0;
      --t.seconds;
    }

    if (lhs.seconds < rhs.seconds || t.seconds <= -60) {
      t.seconds += 60;
      --t.minutes;
    }

    if (lhs.minutes < rhs.minutes || t.minutes <= -60) {
      t.minutes += 60;
      --t.hours;
    }

    return t;
  }

  ////////////////////////////////////////////////////////////////////////////////
  time_duration operator+(time_duration const& lhs, time_duration const& rhs)
  {
    time_duration t;

    t.fractional_seconds  = lhs.fractional_seconds  + rhs.fractional_seconds;
    t.seconds             = lhs.seconds             + rhs.seconds;
    t.minutes             = lhs.minutes             + rhs.minutes;
    t.hours               = lhs.hours               + rhs.hours;

    if (lhs.fractional_seconds + rhs.fractional_seconds >= 1.0) {
      t.fractional_seconds -= 1.0;
      ++t.seconds;
    }

    if (lhs.seconds + rhs.seconds >= 60) {
      t.seconds -= 60;
      ++t.minutes;
    }

    if (lhs.minutes + rhs.minutes >= 60) {
      t.minutes -= 60;
      ++t.hours;
    }

    return t;
  }

  ////////////////////////////////////////////////////////////////////////////////
  time_duration operator/(time_duration const& lhs, double rhs)
  {
    double        secs = lhs.fractional_seconds + lhs.seconds + lhs.minutes * 60.0 + lhs.hours * 3600.0;
    secs              /= rhs;

    time_duration t;

#ifndef WIN32
    typedef int64_t LONGLONG;
#endif
    t.fractional_seconds = double   ( std::fmod  ( secs, 1.0    ));
    t.seconds            = LONGLONG ( std::fmod  ( secs, 60.0   ));
    t.minutes            = LONGLONG ( std::fmod  ( secs, 3600.0 ));
    t.hours              = LONGLONG ( std::floor ( secs / 3600.0));

    return t;
  }

  ////////////////////////////////////////////////////////////////////////////////
  std::ostream& operator<<(std::ostream& os, time_duration const& t)
  {
    os << t.hours << " h, " << t.minutes << " min, " << t.seconds << " s, " << t.fractional_seconds * 1000.0 << " ms";
    return os;
  }




  ////////////////////////////////////////////////////////////////////////////////
  timer::timer()
    : _framecount(0)
  {
    time(_last_frame_integration);

    time(_starttime);
    time(_stoptime);
  }


  ////////////////////////////////////////////////////////////////////////////////
  timer::~timer()
  {}


  ////////////////////////////////////////////////////////////////////////////////
  void
  timer::frame ( std::ostream& os, double integration_time_in_s)
  {
    ++_framecount;

    time_duration current_time;
    time(current_time);

    time_duration duration = current_time - _last_frame_integration;
    double diff_in_s =  duration.hours * 3600.0 +
                        duration.minutes * 60.0 +
                        duration.seconds        +
                        duration.fractional_seconds;

    if (diff_in_s >= integration_time_in_s)
    {
      os << "fps: " << float(_framecount) / diff_in_s << std::endl;
      _last_frame_integration = current_time;
      _framecount = 0;
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  void
  timer::time( time_duration& time )
  {
  #ifdef WIN32
    LARGE_INTEGER ticksPerSecond;
    LARGE_INTEGER tick;
    LARGE_INTEGER current;

    QueryPerformanceFrequency ( &ticksPerSecond );

    DWORD_PTR oldmask = ::SetThreadAffinityMask(::GetCurrentThread(), 0);
    QueryPerformanceCounter   ( &tick );
    SetThreadAffinityMask(::GetCurrentThread(), oldmask);

    current.QuadPart        = tick.QuadPart/ticksPerSecond.QuadPart;
    time.hours              = LONGLONG(current.QuadPart/3600);

    current.QuadPart        = current.QuadPart - (time.hours * 3600);
    time.minutes            = LONGLONG(current.QuadPart / 60);

    time.seconds            = LONGLONG(current.QuadPart - (time.minutes * 60));

    time.fractional_seconds = double(tick.QuadPart % ticksPerSecond.QuadPart)  / ticksPerSecond.QuadPart;
  #else
    timeval tv;
    gettimeofday(&tv ,0);
  #endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  void
  timer::start()
  {
    time(_starttime);
  }

  ////////////////////////////////////////////////////////////////////////////////
  void
  timer::stop()
  {
    time(_stoptime);
  }

  ////////////////////////////////////////////////////////////////////////////////
  time_duration
  timer::result() const
  {
    return _stoptime - _starttime;

  }


} } // namespace gpucast / namespace gl

