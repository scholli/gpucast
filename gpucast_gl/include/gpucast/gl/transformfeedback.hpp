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
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/arraybuffer.hpp>



namespace gpucast { namespace gl {

class GPUCAST_GL transform_feedback
{
public :
  
  transform_feedback();
  ~transform_feedback();

  void            swap         (transform_feedback& );

public : // methods

  bool                        active() const;
  GLenum                      captured_primitive_type() const;

  /* virtual */ void          bind   ( ) const;
  /* virtual */ void          unbind ( ) const;

  void                        begin(GLenum primitive_type);
  void                        end();

  unsigned                    id() const;

private : // members

  unsigned                    _gl_object_id;
  int                         _gl_object_target;
  int                         _gl_object_binding;
  bool                        _active;
  GLenum                      _primitive_type;

};

struct transform_feedback_buffer {
  std::shared_ptr<gpucast::gl::transform_feedback> feedback;
  std::shared_ptr<gpucast::gl::vertexarrayobject>  vertex_array_object;
  std::shared_ptr<gpucast::gl::arraybuffer>        buffer;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_ARRAY_BUFFER_HPP
