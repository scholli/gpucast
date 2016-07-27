/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : transformfeedback.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_TRANSFORMFEEDBACK_BUFFER_HPP
#define GPUCAST_GL_TRANSFORMFEEDBACK_BUFFER_HPP

// header system
#include <string>
#include <memory>
#include <vector>
#include <cassert>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>


namespace gpucast { namespace gl {

  enum primitive_type
  {
    PRIMITIVE_POINTS = 0x00,
    PRIMITIVE_LINES,
    PRIMITIVE_TRIANGLES,
    PRIMITIVE_TYPE_COUNT
  }; // enum primitive_type

  enum primitive_topology {
    PRIMITIVE_POINT_LIST = 0x00,
    PRIMITIVE_LINE_LIST,
    PRIMITIVE_LINE_STRIP,
    PRIMITIVE_LINE_LOOP,
    PRIMITIVE_LINE_LIST_ADJACENCY,
    PRIMITIVE_LINE_STRIP_ADJACENCY,
    PRIMITIVE_TRIANGLE_LIST,
    PRIMITIVE_TRIANGLE_STRIP,
    PRIMITIVE_TRIANGLE_LIST_ADJACENCY,
    PRIMITIVE_TRIANGLE_STRIP_ADJACENCY,
    PRIMITIVE_PATCH_LIST_1_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_2_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_3_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_4_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_5_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_6_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_7_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_8_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_9_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_10_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_11_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_12_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_13_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_14_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_15_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_16_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_17_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_18_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_19_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_20_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_21_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_22_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_23_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_24_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_25_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_26_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_27_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_28_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_29_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_30_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_31_CONTROL_POINTS,
    PRIMITIVE_PATCH_LIST_32_CONTROL_POINTS,
    PRIMITIVE_TOPOLOGY_COUNT
  }; // enum primitive_topology


  GPUCAST_GL unsigned gl_primitive_type(const primitive_type p);

  class GPUCAST_GL stream_output_setup
  {
  public:

    typedef std::pair<buffer_ptr, size_t>   element;
    typedef std::vector<element>            element_array;

  public:
    stream_output_setup() = default;
    stream_output_setup(const element& in_element);
    stream_output_setup(const buffer_ptr& out_buffer, const size_t out_buffer_offset = 0);

    ~stream_output_setup() = default;

    stream_output_setup&    operator()(const element& in_element);
    stream_output_setup&    operator()(const buffer_ptr& out_buffer, const size_t out_buffer_offset = 0);

    void                    insert(const element& in_element);
    void                    insert(const buffer_ptr& out_buffer, const size_t out_buffer_offset = 0);

    bool                    empty() const;
    int                     used_streams() const;

    const element&          operator[](const int stream) const;

    bool                    operator==(const stream_output_setup& rhs) const;
    bool                    operator!=(const stream_output_setup& rhs) const;

  protected:
    element_array           _elements;

  }; // class stream_output_setup



class GPUCAST_GL transform_feedback
{
public :
  
  transform_feedback(const stream_output_setup& in_setup);
  ~transform_feedback();

  void            swap         (transform_feedback& );

public : // methods

  const buffer_ptr&           stream_out_buffer(const int stream) const;
  const buffer_ptr&           operator[](const int stream) const;
  const stream_output_setup&  stream_out_setup() const;

  bool                        active() const;
  primitive_type              captured_topology() const;

  /* virtual */ void          bind   ( ) const;
  /* virtual */ void          unbind ( ) const;

  void                        begin(primitive_type in_topology_mode);
  void                        end();

  bool                        initialize_transform_feedback_object();
  void                        bind_stream_out_buffers() const;
  void                        unbind_stream_out_buffers() const;

private : // members

  unsigned                    _gl_object_id;
  int                         _gl_object_target;
  int                         _gl_object_binding;
  stream_output_setup         _stream_out_setup;
  bool                        _active;
  primitive_type              _captured_topology;

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_ARRAY_BUFFER_HPP
