/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : texturebuffer.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_TEXTURE_BUFFER_HPP
#define GPUCAST_GL_TEXTURE_BUFFER_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/buffer.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL texturebuffer : public buffer
{
public :
  
  texturebuffer                   ();

  texturebuffer                   ( std::size_t bytes,
                                    GLenum      usage = GL_STATIC_DRAW,
                                    GLenum      texture_format = GL_RGBA32F_ARB );

  ~texturebuffer                  ( );

  void            swap            ( texturebuffer& );

public : // methods

  // set texel format for buffer
  void                format     ( GLenum format );
  GLenum              format     ( ) const;

  // bind and unbind texture buffer
  /* virtual */ void  bind       ( ) const;
  /* virtual */ void  unbind     ( ) const;
  void                unbind     ( );

  void                bind       ( GLint texunit );
  void                bind_as_image ( GLuint imageunit, GLint level, GLboolean layered, GLint layer, GLenum access, GLint format );
  
  
  
  GLuint              texid      ( ) const;

  GLenum              target     ( ) const;
 
private : // members

  GLuint          _texid;
  GLint           _unit;
  GLenum          _format;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_TEXTURE_BUFFER_HPP
