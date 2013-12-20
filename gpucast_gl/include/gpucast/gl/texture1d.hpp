/********************************************************************************
*
* Copyright (C) 1009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : texture1d.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TEXTURE1D_HPP
#define GPUCAST_GL_TEXTURE1D_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/texture.hpp>

#include <boost/noncopyable.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL texture1d : public texture, public boost::noncopyable
  {
  public :

    texture1d                  ( );



    texture1d                  ( std::string const& filename );

    texture1d                  ( std::size_t  width,
                                 GLint        internal_format,
                                 GLenum       format,
                                 GLenum       value_type,
                                 GLint        level = 0,
                                 GLboolean    border = 0 );

    ~texture1d                 ( );

    void            swap       ( texture1d& );


  public : // methods

    bool            load       ( std::string const& in_image_path );

    void            resize     ( std::size_t  width,
                                 GLint        internal_format );

    void            teximage   ( GLint    level,
                                 GLint    internal_format,
			                           GLsizei  width,
			                           GLint    border,
			                           GLenum   format,
			                           GLenum   type,
			                           GLvoid*  pixels );

    void            bind       ( );
    void            bind       ( GLint texunit );
    void            unbind     ( );

    void            set_parameteri  ( GLenum pname, GLint value );
    void            set_parameterf  ( GLenum pname, GLfloat value );
    void            set_parameterfv ( GLenum pname, GLfloat const* value );

    GLuint const    id         ( ) const;
    static GLenum   target     ( );

  private : // member

    GLuint          _id;
    GLint           _unit;

    GLsizei         _width;
};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_TEXTURE1D_HPP
