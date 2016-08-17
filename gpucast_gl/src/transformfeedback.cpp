/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : transformfeedback.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/transformfeedback.hpp"

// header system
#include <vector>
#include <iostream>
#include <exception>

// header dependencies
#include <GL/glew.h>

// header project
#include <gpucast/math/vec4.hpp>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  transform_feedback::transform_feedback()
    : _gl_object_id(-1)
    , _gl_object_target(GL_TRANSFORM_FEEDBACK)
    , _gl_object_binding(GL_TRANSFORM_FEEDBACK_BINDING)
    , _active(false)
  {
    glGenTransformFeedbacks(1, &(_gl_object_id));
    if (0 == _gl_object_id) {
      throw std::runtime_error("transform_feedback::transform_feedback(): Unable to initialize transformfeedback.");
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  transform_feedback::~transform_feedback() {
    glDeleteTransformFeedbacks(1, &_gl_object_id);
  }

  ///////////////////////////////////////////////////////////////////////////////
  bool transform_feedback::active() const
  {
    return _active;
  }

  ///////////////////////////////////////////////////////////////////////////////
  GLenum transform_feedback::captured_primitive_type() const
  {
    return _primitive_type;
  }

  ///////////////////////////////////////////////////////////////////////////////
  void transform_feedback::bind() const
  {
    assert(_gl_object_id != 0);
    glBindTransformFeedback(_gl_object_target, _gl_object_id);
  }

  ///////////////////////////////////////////////////////////////////////////////
  void transform_feedback::unbind() const
  {
    assert(_gl_object_id != 0);
    glBindTransformFeedback(_gl_object_target, 0);
  }

  ///////////////////////////////////////////////////////////////////////////////
  void transform_feedback::begin(GLenum primitive_type)
  {
    bind();

    if (!_active) {
      glBeginTransformFeedback(primitive_type);
    }
    
    _active = true;
    _primitive_type = primitive_type;
  }

  ///////////////////////////////////////////////////////////////////////////////
  void transform_feedback::end()
  {
    if (_active) {
      glEndTransformFeedback();
    }

    unbind();
    _active = false;
  }

  ///////////////////////////////////////////////////////////////////////////////
  unsigned transform_feedback::id() const
  {
    return _gl_object_id;
  }

} } // namespace gpucast / namespace gl
