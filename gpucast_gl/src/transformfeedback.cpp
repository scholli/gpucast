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
  stream_output_setup::stream_output_setup(const element& in_element){
    insert(in_element);
  }

  ///////////////////////////////////////////////////////////////////////////////
  stream_output_setup::stream_output_setup(const buffer_ptr& out_buffer, const size_t out_buffer_offset)
  {
    insert(out_buffer, out_buffer_offset);
  }
  ///////////////////////////////////////////////////////////////////////////////
  stream_output_setup& stream_output_setup::operator()(const element& in_element)
  {
    insert(in_element);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////
  stream_output_setup&
    stream_output_setup::operator()(const buffer_ptr& out_buffer, const size_t out_buffer_offset)
  {
    insert(out_buffer, out_buffer_offset);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////
  void stream_output_setup::insert(const element& in_element)
  {
    _elements.push_back(in_element);
  }

  ///////////////////////////////////////////////////////////////////////////////
  void stream_output_setup::insert(const buffer_ptr& out_buffer, const size_t out_buffer_offset)
  {
    _elements.push_back(element(out_buffer, out_buffer_offset));
  }

  ///////////////////////////////////////////////////////////////////////////////
  int stream_output_setup::used_streams() const
  {
    assert(_elements.size() < (std::numeric_limits<int>::max)());
    return static_cast<int>(_elements.size());
  }

  ///////////////////////////////////////////////////////////////////////////////
  bool stream_output_setup::empty() const
  {
    return _elements.empty();
  }

  ///////////////////////////////////////////////////////////////////////////////
  const stream_output_setup::element& stream_output_setup::operator[](const int stream) const {
    assert(0 <= stream && stream < _elements.size());
    return _elements[stream];
  }

  ///////////////////////////////////////////////////////////////////////////////
  bool stream_output_setup::operator==(const stream_output_setup& rhs) const
  {
    return _elements == rhs._elements;
  }

  ///////////////////////////////////////////////////////////////////////////////
  bool stream_output_setup::operator!=(const stream_output_setup& rhs) const
  {
    return _elements != rhs._elements;
  }

  ///////////////////////////////////////////////////////////////////////////////
  transform_feedback::transform_feedback(const stream_output_setup&   in_setup)
    : _gl_object_id(-1)
    , _gl_object_target(GL_TRANSFORM_FEEDBACK)
    , _gl_object_binding(GL_TRANSFORM_FEEDBACK_BINDING)
    , _stream_out_setup(in_setup)
    , _active(false)
    , _captured_topology(PRIMITIVE_POINTS)
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
  const buffer_ptr& transform_feedback::stream_out_buffer(const int stream) const
  {
    assert(0 <= stream && stream < _stream_out_setup.used_streams());
    return _stream_out_setup[stream].first;
  }

  ///////////////////////////////////////////////////////////////////////////////
  const buffer_ptr& transform_feedback::operator[](const int stream) const
  {
    return stream_out_buffer(stream);
  }

  ///////////////////////////////////////////////////////////////////////////////
  const stream_output_setup& transform_feedback::stream_out_setup() const
  {
    return _stream_out_setup;
  }

  ///////////////////////////////////////////////////////////////////////////////
  bool transform_feedback::active() const
  {
    return _active;
  }

  ///////////////////////////////////////////////////////////////////////////////
  primitive_type transform_feedback::captured_topology() const
  {
    return _captured_topology;
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
  void transform_feedback::begin(primitive_type in_topology_mode)
  {
    bind();

    if (!_active) {
      glBeginTransformFeedback(gl_primitive_type(in_topology_mode));
    }
    
    _active = true;
    _captured_topology = in_topology_mode;
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
  bool transform_feedback::initialize_transform_feedback_object()
  {
    if (_stream_out_setup.empty()) {
      return false;
    }

    if (_stream_out_setup.used_streams() > GL_MAX_TRANSFORM_FEEDBACK_BUFFERS) {
      return false;
    }
    
    bind();
    bind_stream_out_buffers();

    return true;
  }

  ///////////////////////////////////////////////////////////////////////////////
  void transform_feedback::bind_stream_out_buffers() const
  {
    for (int bind_index = 0; bind_index < _stream_out_setup.used_streams(); ++bind_index) {
      const buffer_ptr& cur_buffer = _stream_out_setup[bind_index].first;
      const size_t      cur_offset = _stream_out_setup[bind_index].second;

      if (cur_buffer) {
        cur_buffer->bind_range(bind_index, cur_offset, 0);
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  void transform_feedback::unbind_stream_out_buffers() const
  {
    for (int bind_index = 0; bind_index < _stream_out_setup.used_streams(); ++bind_index) {
      const buffer_ptr& cur_buffer = _stream_out_setup[bind_index].first;

      if (cur_buffer) {
        cur_buffer->unbind_range(bind_index);
      }
    }
  }

} } // namespace gpucast / namespace gl
