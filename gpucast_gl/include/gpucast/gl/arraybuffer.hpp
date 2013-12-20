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
#ifndef GPUCAST_GL_ARRAY_BUFFER_HPP
#define GPUCAST_GL_ARRAY_BUFFER_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL arraybuffer : public buffer
{
public :
  
  arraybuffer                  ( );

  arraybuffer                  ( std::size_t bytes, GLenum usage = GL_STATIC_DRAW );

  ~arraybuffer                 ( );

  void            swap         ( arraybuffer& );

public : // methods

  /* virtual */ void    bind   ( ) const;

  /* virtual */ void    unbind ( ) const;

  /* virtual */ GLenum  target ( ) const;
 
private : // members

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_ARRAY_BUFFER_HPP
