/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : texture3d.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TEXTURE3D_HPP
#define GPUCAST_GL_TEXTURE3D_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/texture.hpp>

#include <boost/noncopyable.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL texture3d : public texture, public boost::noncopyable
  {
  public :

    texture3d                  ( );



    texture3d                  ( std::string const& filename );

    texture3d                  ( std::size_t  width,
                                 std::size_t  height,
                                 std::size_t  depth,
                                 GLint        internal_format,
                                 GLenum       format,
                                 GLenum       value_type,
                                 GLint        level = 0,
                                 GLboolean    border = 0 );

    ~texture3d                 ( );

    void            swap       ( texture3d& );


  public : // methods

    bool            load       ( std::string const& filename, unsigned depth );

    void            resize     ( std::size_t width,
                                 std::size_t height,
                                 std::size_t depth,
                                 GLint internal_format );

    void            teximage   ( GLint    level,
                                 GLint    internal_format,
			                           GLsizei  width,
			                           GLsizei  height,
                                 GLsizei  depth, 
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
    GLsizei         _height;
    GLsizei         _depth;
};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_TEXTURE3D_HPP
