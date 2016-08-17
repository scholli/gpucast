/********************************************************************************
*
* Copyright (C) 2015 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : transformfeedback_query.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TRANSFORMFEEDBACK_QUERY_HPP
#define GPUCAST_GL_TRANSFORMFEEDBACK_QUERY_HPP

// header system
#include <string>
#include <map>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/query.hpp>

namespace gpucast {
  namespace gl {

    class GPUCAST_GL transformfeedback_query : public query
    {
    public: // methods

      /*virtual*/ void begin() const override;
      /*virtual*/ void end() const override;

      long long primitives_written(bool wait = true) const; // in ms

    private: // member

    };

  }
} // namespace gpucast / namespace gl


#endif // GPUCAST_GL_TRANSFORMFEEDBACK_QUERY_HPP
