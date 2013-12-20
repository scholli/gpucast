/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : texturearray.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TEXTUREARRAYHPP
#define GPUCAST_GL_TEXTUREARRAYHPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>

#include <boost/noncopyable.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL texturearray : boost::noncopyable
  {
  public :

    texturearray                  ( );

    texturearray                  ( std::size_t width,
                                    std::size_t height,
                                    std::size_t depth );

    texturearray                  ( std::size_t width,
                                    std::size_t height,
                                    std::size_t depth,
                                    int internal_format,
                                    int format,
                                    int value_type,
                                    int level = 0,
                                    bool border = 0 );

    ~texturearray                  ( );

    void            swap           ( texturearray& );

  public : // methods

    void            teximage       ( GLint    level,
                                     GLint    internal_format,
			                               GLsizei  width,
			                               GLsizei  height,
                                     GLsizei  depth,
			                               GLint    border,
			                               GLenum   format,
			                               GLenum   type,
			                               GLvoid*  pixels );

    void            texsubimage    ( GLint    level,
                                     GLint  	xoffset, 
 	                                   GLint  	yoffset, 
 	                                   GLint  	zoffset, 
			                               GLsizei  width,
			                               GLsizei  height,
                                     GLsizei  depth,
			                               GLenum   format,
			                               GLenum   type,
			                               GLvoid*  data );

    void            bind            ( );
    void            bind            ( GLint texunit );
    void            unbind          ( );

    void            set_parameteri  ( GLenum pname, GLint value );
    void            set_parameterf  ( GLenum pname, GLfloat value );
    void            set_parameterfv ( GLenum pname, GLfloat const* value );

    GLuint const    id           ( ) const;
    static GLenum   target       ( );

  private : // member

    GLuint          _id;
    GLint           _unit;

    std::size_t     _width;
    std::size_t     _height;
    std::size_t     _depth;
};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_TEXTURE2D_HPP
