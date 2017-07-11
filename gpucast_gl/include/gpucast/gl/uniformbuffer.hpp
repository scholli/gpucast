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
#ifndef GPUCAST_GL_UNIFORM_BUFFER_HPP
#define GPUCAST_GL_UNIFORM_BUFFER_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL uniformbuffer : public buffer
{
public :
  
  uniformbuffer                  ( );

  uniformbuffer                  ( std::size_t bytes, GLenum usage );

  ~uniformbuffer                 ( );

  uniformbuffer& operator=       ( uniformbuffer const& );

  void            swap           ( uniformbuffer& );

public : // methods

  // bind and unbind arraybuffer

  /* virtual */ void    bind     ( ) const;
  /* virtual */ void    unbind   ( ) const;

  void                  bindbase ( GLuint base ) const;

  /* virtual */ GLenum  target   () const;
  
private : // attibutes

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_UNIFORM_BUFFER_HPP
