/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : atomicbuffer.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_ATOMIC_BUFFER_HPP
#define GPUCAST_GL_ATOMIC_BUFFER_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>


namespace gpucast { namespace gl {

  class GPUCAST_GL atomicbuffer : public buffer
{
public :
  
  atomicbuffer();

  atomicbuffer(std::size_t bytes, GLenum usage = GL_STATIC_DRAW);

  ~atomicbuffer();

  void            swap(atomicbuffer&);

public : // methods

  /* virtual */ void    bind   ( ) const;

  void bind_buffer_range(unsigned binding_point, unsigned offset, unsigned count) const;
  void bind_buffer_base(unsigned binding_point) const;

  /* virtual */ void    unbind ( ) const;

  /* virtual */ GLenum  target ( ) const;
 
private : // members

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_ATOMIC_BUFFER_HPP
