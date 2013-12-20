/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : elementarraybuffer.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_ELEMENT_ARRAY_HPP
#define GPUCAST_GL_ELEMENT_ARRAY_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL elementarraybuffer : public buffer 
{
public :
  
  elementarraybuffer                  ( );

  elementarraybuffer                  ( std::size_t bytes, GLenum usage );

  ~elementarraybuffer                 ( );
  
  void            swap                ( elementarraybuffer& );

public : // methods


  /* virtual */ void    bind          ( ) const;
  
  /* virtual */ void    unbind        ( ) const;

  /* virtual */ GLenum  target        ( ) const;
 

private : // attributes

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_ELEMENT_ARRAY_HPP
