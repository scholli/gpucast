/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : framebufferobject.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_FRAMEBUFFER_OBJECT_HPP
#define GPUCAST_GL_FRAMEBUFFER_OBJECT_HPP

#include <iostream>
#include <GL/glew.h>

#include <gpucast/gl/glpp.hpp>
#include <boost/noncopyable.hpp>


namespace gpucast {
  namespace gl {

    // forward declaration
    class renderbuffer;
    class texture2d;

    class GPUCAST_GL framebufferobject : public boost::noncopyable
    {
    public:

      framebufferobject();
      ~framebufferobject();

    public: // method

      bool              status() const;
      bool              is_framebuffer() const;
      void              print(std::ostream& os) const;

      void              bind() const;
      void              unbind() const;
      bool              bound() const;

      void              attach_texture(texture2d&   texture,
        GLenum       attachment,
        int          miplevel = 0) const;

      void              attach_renderbuffer(renderbuffer const&  renderbuf,
        GLenum               attachment) const;

      void              unattach(GLenum attachment) const;

      static GLuint     max_color_attachments();
      static GLenum     target();

    private:

      unsigned _id;
    };





#if 0
    class GPUCAST_GL framebufferobject
    {
    public :
      framebufferobject();
      virtual ~framebufferobject();

    public :
      void Bind();

      /// Bind a texture to the "attachment" point of this FBO
      virtual void AttachTexture( GLenum texTarget,
        GLuint texId,
        GLenum attachment = GL_COLOR_ATTACHMENT0_EXT,
        int mipLevel      = 0,
        int zSlice        = 0 );

      /// Bind an array of textures to multiple "attachment" points of this FBO
      ///  - By default, the first 'numTextures' attachments are used,
      ///    starting with GL_COLOR_ATTACHMENT0_EXT
      virtual void AttachTextures( int numTextures,
        GLenum texTarget[],
        GLuint texId[],
        GLenum attachment[] = NULL,
        int mipLevel[]      = NULL,
        int zSlice[]        = NULL );

      /// Bind a render buffer to the "attachment" point of this FBO
      virtual void AttachRenderBuffer( GLuint buffId,
        GLenum attachment = GL_COLOR_ATTACHMENT0_EXT );

      /// Bind an array of render buffers to corresponding "attachment" points
      /// of this FBO.
      /// - By default, the first 'numBuffers' attachments are used,
      ///   starting with GL_COLOR_ATTACHMENT0_EXT
      virtual void AttachRenderBuffers( int numBuffers, GLuint buffId[],
        GLenum attachment[] = NULL );

      /// Free any resource bound to the "attachment" point of this FBO
      void Unattach( GLenum attachment );

      /// Free any resources bound to any attachment points of this FBO
      void UnattachAll();

      bool valid();

      GLenum GetAttachedType( GLenum attachment );
      GLuint GetAttachedId( GLenum attachment );
      GLint  GetAttachedMipLevel( GLenum attachment );
      GLint  GetAttachedCubeFace( GLenum attachment );
      GLint  GetAttachedZSlice( GLenum attachment );

      static int GetMaxColorAttachments();
      static void Disable();

    protected:
      void  _GuardedBind();
      void  _GuardedUnbind();
      void  _FramebufferTextureND( GLenum attachment, GLenum texTarget,
        GLuint texId, int mipLevel, int zSlice );
      static GLuint _GenerateFboId();

    private:
      GLuint m_fboId;
      GLuint m_savedFboId;


    };
#endif

} } // namespace gpucast / namespace gl

#endif

