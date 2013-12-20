/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : renderbuffer.hpp
*  project    : glpp
*  description:
*
********************************************************************************/

#ifndef GPUCAST_GL_RENDER_BUFFER_HPP
#define GPUCAST_GL_RENDER_BUFFER_HPP

// i/f includes
#include <gpucast/gl/framebufferobject.hpp>
#include <boost/noncopyable.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL renderbuffer : public boost::noncopyable
  {
    public:

      static GLenum const TARGET = GL_RENDERBUFFER_EXT;

      renderbuffer();
      renderbuffer(GLenum internal_format, std::size_t width, std::size_t height);
      ~renderbuffer();

    public :

      void            bind    ( ) const;
      void            unbind  ( ) const;
      bool            bound   ( ) const;

      void            set     ( GLenum internal_format, std::size_t width, std::size_t height) const;
      GLuint          id      ( ) const;

      void            print   ( std::ostream& os ) const;

      static GLint    get_maxsize();

      GLenum          target  () const;

  private:

      unsigned _id;
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_RENDER_BUFFER_HPP

