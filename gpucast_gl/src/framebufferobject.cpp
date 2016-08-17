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



#if 0

///////////////////////////////////////////////////////////////////////////////
framebufferobject::framebufferobject()
  : m_fboId(_GenerateFboId()),
    m_savedFboId(0)
{
  // Bind this FBO so that it actually gets created now
  _GuardedBind();
  _GuardedUnbind();
}


///////////////////////////////////////////////////////////////////////////////
framebufferobject::~framebufferobject()
{
  glDeleteFramebuffersEXT(1, &m_fboId);
}


///////////////////////////////////////////////////////////////////////////////
void framebufferobject::Bind()
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fboId);
}


///////////////////////////////////////////////////////////////////////////////
void framebufferobject::Disable()
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::AttachTexture( GLenum texTarget, GLuint texId,
                                  GLenum attachment, int mipLevel, int zSlice )
{
  _GuardedBind();
  glActiveTexture(GL_TEXTURE0 + texId);
  _FramebufferTextureND( attachment, texTarget, texId, mipLevel, zSlice );
  _GuardedUnbind();
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::AttachTextures( int numTextures, GLenum texTarget[], GLuint texId[],
                                  GLenum attachment[], int mipLevel[], int zSlice[] )
{
  for(int i = 0; i < numTextures; ++i)
  {
    AttachTexture( texTarget[i], texId[i],
                   attachment ? attachment[i] : (GL_COLOR_ATTACHMENT0_EXT + i),
                   mipLevel ? mipLevel[i] : 0,
                   zSlice ? zSlice[i] : 0 );
  }
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::AttachRenderBuffer( GLuint buffId, GLenum attachment )
{
  _GuardedBind();
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, buffId);
  _GuardedUnbind();
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::AttachRenderBuffers( int numBuffers, GLuint buffId[], GLenum attachment[] )
{
  for(int i = 0; i < numBuffers; ++i)
  {
    AttachRenderBuffer( buffId[i], attachment ? attachment[i] : (GL_COLOR_ATTACHMENT0_EXT + i) );
  }
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::Unattach( GLenum attachment )
{
  _GuardedBind();
  GLenum type = GetAttachedType(attachment);

  switch(type) {
  case GL_NONE:
    break;
  case GL_RENDERBUFFER_EXT:
    AttachRenderBuffer( 0, attachment );
    break;
  case GL_TEXTURE_2D:
    AttachTexture( GL_TEXTURE_2D, 0, attachment );
    break;
  default:
    BOOST_LOG_TRIVIAL(error) << "framebufferobject::unbind_attachment ERROR: Unknown attached resource type\n";
  }
  _GuardedUnbind();
}


///////////////////////////////////////////////////////////////////////////////
bool
framebufferobject::valid()
{
  GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

	switch (status)
		{
		case GL_FRAMEBUFFER_COMPLETE_EXT:
			BOOST_LOG_TRIVIAL(error) << "GL_FRAMEBUFFER_COMPLETE_EXT" << std::endl;
			return true;
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
			BOOST_LOG_TRIVIAL(error) << "[ERROR] : unknown fbo error" << std::endl;
			break;
	}
	return false;
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::UnattachAll()
{
  int numAttachments = GetMaxColorAttachments();
  for(int i = 0; i < numAttachments; ++i)
  {
    Unattach( GL_COLOR_ATTACHMENT0_EXT + i );
  }
}


///////////////////////////////////////////////////////////////////////////////
GLint
framebufferobject::GetMaxColorAttachments()
{
  GLint maxAttach = 0;
  glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS, &maxAttach );
  return maxAttach;
}


///////////////////////////////////////////////////////////////////////////////
GLuint
framebufferobject::_GenerateFboId()
{
  GLuint id = 0;
  glGenFramebuffersEXT(1, &id);
  return id;
}

///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::_GuardedBind()
{
  // Only binds if m_fboId is different than the currently bound FBO
  GLint binding;
  glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &binding );
  m_savedFboId = binding;
  if (m_fboId != m_savedFboId)
  {
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fboId);
  }
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::_GuardedUnbind()
{
  // Returns FBO binding to the previously enabled FBO
  if (m_fboId != m_savedFboId)
   {
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_savedFboId);
  }
}


///////////////////////////////////////////////////////////////////////////////
void
framebufferobject::_FramebufferTextureND( GLenum attachment, GLenum texTarget,
                                          GLuint texId, int mipLevel,
                                          int zSlice )
{
  if (texTarget == GL_TEXTURE_1D)
  {
    glFramebufferTexture1DEXT( GL_FRAMEBUFFER_EXT, attachment, GL_TEXTURE_1D, texId, mipLevel );
  }
  else
  {
    if (texTarget == GL_TEXTURE_3D)
    {
      glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, attachment, GL_TEXTURE_3D, texId, mipLevel, zSlice );
    }
    else
    {
      // Default is GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_ARB, or cube faces
      glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, attachment, texTarget, texId, mipLevel );
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
GLenum
framebufferobject::GetAttachedType( GLenum attachment )
{
  // Returns GL_RENDERBUFFER_EXT or GL_TEXTURE
  _GuardedBind();
  GLint type = 0;
  glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_EXT, &type);
  _GuardedUnbind();
  return GLenum(type);
}


///////////////////////////////////////////////////////////////////////////////
GLuint
framebufferobject::GetAttachedId( GLenum attachment )
{
  _GuardedBind();
  GLint id = 0;
  glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT, &id);
  _GuardedUnbind();
  return GLuint(id);
}


///////////////////////////////////////////////////////////////////////////////
GLint
framebufferobject::GetAttachedMipLevel( GLenum attachment )
{
  _GuardedBind();
  GLint level = 0;
  glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL_EXT, &level);
  _GuardedUnbind();
  return level;
}

///////////////////////////////////////////////////////////////////////////////
GLint
framebufferobject::GetAttachedCubeFace( GLenum attachment )
{
  _GuardedBind();
  GLint level = 0;
  glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE_EXT, &level);
  _GuardedUnbind();
  return level;
}


///////////////////////////////////////////////////////////////////////////////
GLint
framebufferobject::GetAttachedZSlice( GLenum attachment )
{
  _GuardedBind();
  GLint slice = 0;
  glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_EXT, &slice);
  _GuardedUnbind();
  return slice;
}
#endif

} } // namespace gpucast / namespace gl
