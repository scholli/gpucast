/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : arraybuffer.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_SHADER_STORAGE_BUFFER_HPP
#define GPUCAST_GL_SHADER_STORAGE_BUFFER_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL shaderstoragebuffer : public buffer
{
public :
  
  shaderstoragebuffer();

  shaderstoragebuffer(std::size_t bytes, GLenum usage = GL_STATIC_DRAW);

  ~shaderstoragebuffer();

  void            swap(shaderstoragebuffer&);

public : // methods

  /* virtual */ void    bind   ( ) const;

  /* virtual */ void    unbind ( ) const;

  /* virtual */ GLenum  target ( ) const;
 
private : // members

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_SHADER_STORAGE_BUFFER_HPP
