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
#ifndef GPUCAST_GL_TEXTURE_HPP
#define GPUCAST_GL_TEXTURE_HPP

// header system
#include <string>
#include <map>

// header project
#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  struct GPUCAST_GL texture_format 
  {
    texture_format();
    texture_format(GLenum internal_fmt, GLenum base_fmt, std::size_t components_size, GLenum value_type, std::string const& as_string);
    GLenum        internal_format;
    GLenum        base_format;
    std::size_t   size;
    GLenum        type;
    std::string   name;
  };

  class GPUCAST_GL texture
  {
  public :

    texture                                       ( );
    ~texture                                      ( );

    std::size_t             size_of_format        ( GLenum internal_format );
    GLenum                  base_format           ( GLenum internal_format );
    GLenum                  value_type_of_format  ( GLenum internal_format );
    std::string             name                  ( GLenum internal_format );

  public : // methods

    virtual void            bind                  ( GLint texunit ) = 0;
    virtual void            unbind                ( ) = 0;
    virtual GLuint const    id                    ( ) const = 0;

  private : // methods

    void                    _init                 ();

  private : // member

};

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_TEXTURE1D_HPP
