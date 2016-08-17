/********************************************************************************
*
* Copyright (C) 2015 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : transformfeedback_query.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/transformfeedback_query.hpp"
#include "gpucast/gl/error.hpp"

// header system
#include <GL/glew.h>

namespace gpucast {
  namespace gl {

    ////////////////////////////////////////////////////////////////////////////////
    void transformfeedback_query::begin() const {
      glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, id());
    }

    ////////////////////////////////////////////////////////////////////////////////
    void transformfeedback_query::end() const {
      glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    }

    ////////////////////////////////////////////////////////////////////////////////
    long long transformfeedback_query::primitives_written(bool wait) const {
      if (wait) {
        while (is_available() == false) {}
      }
      GLuint64 result = 0;
      glGetQueryObjectui64v(id(), GL_QUERY_RESULT, &result);
      return result;
    }

  }
} // namespace gpucast / namespace gl
