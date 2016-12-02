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
// i/f include
#include "gpucast/gl/framebufferobject.hpp"

// system include
#include <iostream>
#include <cassert>
#include <GL/glew.h>

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/texture2d.hpp>

#include <boost/log/trivial.hpp>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  framebufferobject::framebufferobject()
  : _id(0)
  {
    glGenFramebuffersEXT(1, &_id);
    bind();
    unbind();
  }


  ///////////////////////////////////////////////////////////////////////////////
  framebufferobject::~framebufferobject()
  {
    glDeleteFramebuffersEXT(1, &_id);
  }


  ///////////////////////////////////////////////////////////////////////////////
  bool
  framebufferobject::status() const
  {
    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

    switch (status)
      {
      case GL_FRAMEBUFFER_COMPLETE_EXT:
        BOOST_LOG_TRIVIAL(error) << "Initialize Framebufferobject: ok" << std::endl;
        return true;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT" << std::endl;
        break;
      case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : GL_FRAMEBUFFER_UNSUPPORTED_EXT" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT" << std::endl;
        break;
      //case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
      //	BOOST_LOG_TRIVIAL(error) << "GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT" << std::endl;
      //	break;
      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT" << std::endl;
        break;
      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT" << std::endl;
        break;
      default:
        BOOST_LOG_TRIVIAL(error) << "[ERROR] : UNKNOWN FRAMEBUFFER ERROR" << std::endl;
        break;
    }

    return false;
  }


  ///////////////////////////////////////////////////////////////////////////////
  bool
  framebufferobject::is_framebuffer () const
  {
    return glIsFramebufferEXT(_id) == GL_TRUE;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  framebufferobject::print ( std::ostream& os ) const
  {
    unsigned max_color_attach = max_color_attachments();

    int object_type;
    int object_id;

    for (unsigned i = 0; i < max_color_attach; ++i)
    {
        glGetFramebufferAttachmentParameterivEXT(target(),
                                                 GL_COLOR_ATTACHMENT0_EXT+i,
                                                 GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT,
                                                 &object_type);
        if ( object_type != GL_NONE )
        {
            glGetFramebufferAttachmentParameterivEXT(target(),
                                                     GL_COLOR_ATTACHMENT0_EXT+i,
                                                     GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT,
                                                     &object_id);

            os << "Color Attachment " << i << ": ";
            if (object_type == GL_TEXTURE)
                os << "GL_TEXTURE : " << object_id << std::endl;
            else if (object_type == GL_RENDERBUFFER_EXT)
                os << "GL_RENDERBUFFER_EXT : " << object_id << std::endl;
        }
    }

    // print info of the depthbuffer attachable image
    glGetFramebufferAttachmentParameterivEXT(target(),
                                             GL_DEPTH_ATTACHMENT_EXT,
                                             GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT,
                                             &object_type);
    if(object_type != GL_NONE)
    {
        glGetFramebufferAttachmentParameterivEXT(target(),
                                                 GL_DEPTH_ATTACHMENT_EXT,
                                                 GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT,
                                                 &object_id);

        os << "Depth Attachment: ";
        switch(object_type)
        {
        case GL_TEXTURE:
            os << "GL_TEXTURE :" << object_id << std::endl;
            break;
        case GL_RENDERBUFFER_EXT:
            os << "GL_RENDERBUFFER_EXT :" << object_id << std::endl;
            break;
        }
    }

    // print info of the stencilbuffer attachable image
    glGetFramebufferAttachmentParameterivEXT(target(),
                                             GL_STENCIL_ATTACHMENT_EXT,
                                             GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT,
                                             &object_type);
    if(object_type != GL_NONE)
    {
        glGetFramebufferAttachmentParameterivEXT(target(),
                                                 GL_STENCIL_ATTACHMENT_EXT,
                                                 GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT,
                                                 &object_id);

        os << "Stencil Attachment: ";
        switch(object_type)
        {
        case GL_TEXTURE:
            os << "GL_TEXTURE: " << object_id << std::endl;
            break;
        case GL_RENDERBUFFER_EXT:
            os << "GL_RENDERBUFFER_EXT: " << object_id << std::endl;
            break;
        }
    }
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  framebufferobject::bind () const
  {
    glBindFramebufferEXT(target(), _id);
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  framebufferobject::unbind() const
  {
    glBindFramebufferEXT(target(), 0);
  }


  ///////////////////////////////////////////////////////////////////////////////
  bool
  framebufferobject::bound() const
  {
    int fbo_bound;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &fbo_bound );
    return (int(_id) == fbo_bound);
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  framebufferobject::attach_texture ( texture2d&        tex,
                                      GLenum            attachment,
                                      int               mipmap_level ) const
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glNamedFramebufferTexture2DEXT(_id, attachment, tex.target(), tex.id(), mipmap_level);
#else 
    bind();
    glFramebufferTexture2DEXT( target(), attachment, texture2d::target(), tex.id(), mipmap_level );
    unbind();
#endif
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  framebufferobject::attach_renderbuffer(renderbuffer const& rb, GLenum attachment ) const
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glNamedFramebufferRenderbufferEXT(_id, attachment, rb.target(), rb.id());
#else
    bind();
    glFramebufferRenderbufferEXT( target(), attachment, renderbuffer::target(), rb.id() );
    unbind();
#endif
  }


  ///////////////////////////////////////////////////////////////////////////////
  void
  framebufferobject::unattach(GLenum attachment) const
  {
    bind();

    GLint type = 0;
    glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT, &type);

    switch(type) {
    case GL_NONE:
      break;
    case GL_RENDERBUFFER_EXT:
      glFramebufferTexture2DEXT( target(), attachment, texture2d::target(), 0, 0 );
      break;
    case GL_TEXTURE_2D:
      glFramebufferRenderbufferEXT( target(), attachment, renderbuffer::TARGET, 0);
      break;
    default:
      BOOST_LOG_TRIVIAL(error) << "framebufferobject::unbind_attachment ERROR: Unknown attached resource type\n";
    }

    unbind();
  }


  ///////////////////////////////////////////////////////////////////////////////
  /* static */ GLuint
  framebufferobject::max_color_attachments()
  {
    GLint max = 0;
    glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS, &max );
    return max;
  }


  ///////////////////////////////////////////////////////////////////////////////
  /* static */ GLenum
  framebufferobject::target()
  {
    return GL_FRAMEBUFFER_EXT;
  }

  ///////////////////////////////////////////////////////////////////////////////
  GLuint framebufferobject::id() const
  {
    return _id;
  }

} } // namespace gpucast / namespace gl
