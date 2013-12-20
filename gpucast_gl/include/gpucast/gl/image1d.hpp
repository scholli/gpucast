/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : image1d.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_IMAGE1D_HPP
#define GPUCAST_GL_IMAGE1D_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>

#include <boost/noncopyable.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL image1d : boost::noncopyable
  {
  public :

    image1d                     ( std::size_t width );

    image1d                     ( std::size_t width,
                                    int internal_format,
                                    int format,
                                    int value_type,
                                    int level = 0,
                                    bool border = 0 );

    ~image1d                    ( );

    void            swap        ( image1d& );


  public : // methods

    void            resize      ( std::size_t width );

    void            teximage    ( GLint    level,
                                  GLint    internal_format,
			                            GLsizei  width,
			                            GLint    border,
			                            GLenum   format,
			                            GLenum   type,
			                            GLvoid*  pixels );

    void            bind        ( );
    void            bind        ( GLint texunit );
    void            unbind      ( );

    void            set_parameteri  ( GLenum pname, GLint value );
    void            set_parameterf  ( GLenum pname, GLfloat value );
    void            set_parameterfv ( GLenum pname, GLfloat const* value );

    GLuint const    id         ( ) const;
    static GLenum   target     ( );

  private : // member

    GLuint          _id;
    GLint           _unit;

    std::size_t     _width;
};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_IMAGE1D_HPP
