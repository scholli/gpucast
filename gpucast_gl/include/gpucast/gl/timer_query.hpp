/********************************************************************************
*
* Copyright (C) 2015 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : timer_query.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TIMER_QUERY_HPP
#define GPUCAST_GL_TIMER_QUERY_HPP

// header system
#include <string>
#include <map>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/query.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL timer_query : public query
  {
  public : // methods

    /*virtual*/ void begin() const override;
    /*virtual*/ void end() const override;

    double time_in_ms(bool wait = true) const; // in ms

  private : // member

};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_QUERY_HPP
