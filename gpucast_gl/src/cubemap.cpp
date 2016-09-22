/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : cubemap.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/cubemap.hpp"

#include <cassert>
#include <fstream>
#include <iostream>

#include <GL/glew.h>

// disable boost gil warnings
#if WIN32
  #pragma warning (disable:4714)
  #pragma warning (disable:4127)
  #pragma warning (disable:4512)
#endif

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/texture.hpp>

#include <boost/shared_array.hpp>
#include <boost/log/trivial.hpp>

// header project
#include <gpucast/math/vec3.hpp>

#include <FreeImagePlus.h>

namespace gpucast {
  namespace gl {

    ///////////////////////////////////////////////////////////////////////////////
    cubemap::cubemap()
      : id_(0),
      unit_(-1)
    {
      glGenTextures(1, &id_);
    }

    ///////////////////////////////////////////////////////////////////////////////
    cubemap::~cubemap()
    {
      glDeleteTextures(1, &id_);
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::swap(cubemap& swp)
    {
      std::swap(id_, swp.id_);
      std::swap(unit_, swp.unit_);
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::load(std::string const& imgfile_posx,
        std::string const& imgfile_negx,
        std::string const& imgfile_posy,
        std::string const& imgfile_negy,
        std::string const& imgfile_posz,
        std::string const& imgfile_negz)
    {
      bind();

      openfile_(imgfile_posx, GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT);
      openfile_(imgfile_negx, GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT);

      openfile_(imgfile_posy, GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT);
      openfile_(imgfile_negy, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT);

      openfile_(imgfile_posz, GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT);
      openfile_(imgfile_negz, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT);

      unbind();
    }

    ///////////////////////////////////////////////////////////////////////////////
    GLuint const
    cubemap::id() const
    {
      return id_;
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::bind()
    {
      glBindTexture(cubemap::target(), id_);
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::bind(GLint texunit)
    {
      if (unit_ >= 0) {
        unbind();
      }
      unit_ = texunit;

#if GPUCAST_GL_DIRECT_STATE_ACCESS
      glBindMultiTextureEXT(GL_TEXTURE0 + unit_, cubemap::target(), id_);
#else
      glActiveTexture(GL_TEXTURE0 + unit_);
      enable();
      glBindTexture(cubemap::target(), id_);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::unbind()
    {
      if (unit_ >= 0)
      {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
        glBindMultiTextureEXT(GL_TEXTURE0 + unit_, cubemap::target(), 0);
#else
        glActiveTexture(GL_TEXTURE0 + unit_);
        glBindTexture(cubemap::target(), 0);
        disable();
#endif
      }
      unit_ = -1;
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::enable() const
    {
      glEnable(cubemap::target());
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::disable() const
    {
      glDisable(cubemap::target());
    }

    ///////////////////////////////////////////////////////////////////////////////
    void
      cubemap::parameter(GLenum pname, int param)
    {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
      glTextureParameteriEXT(id_, cubemap::target(), pname, param);
#else
      bind();
      glTexParameteri(cubemap::target(), pname, param);
      unbind();
#endif
    }

    ///////////////////////////////////////////////////////////////////////////////
    /* static */ GLenum
      cubemap::target()
    {
      return GL_TEXTURE_CUBE_MAP;
    }

    ///////////////////////////////////////////////////////////////////////////////
    /* private */ void
      cubemap::openfile_(std::string const& in_image_path, GLenum t)
    {
      std::shared_ptr<fipImage>   in_image(new fipImage);

      if (!in_image->load(in_image_path.c_str()))
      {
        BOOST_LOG_TRIVIAL(error) << "cubemap::openfile_(): " << "unable to open file: " << in_image_path << std::endl;
      }

      FREE_IMAGE_TYPE image_type = in_image->getImageType();
      unsigned        image_width = in_image->getWidth();
      unsigned        image_height = in_image->getHeight();
      GLenum          image_internal_format = 0;
      GLenum          image_format = 0;
      GLenum          image_value_type = 0;

      switch (image_type)
      {
      case FIT_BITMAP:
      {
        unsigned num_components = in_image->getBitsPerPixel() / 8;
        switch (num_components)
        {
        case 1:         image_internal_format = GL_R8;      image_format = GL_RED;  image_value_type = GL_UNSIGNED_BYTE; break;
        case 2:         image_internal_format = GL_RG8;     image_format = GL_RG;   image_value_type = GL_UNSIGNED_BYTE; break;
        case 3:         image_internal_format = GL_RGB8;    image_format = GL_BGR;  image_value_type = GL_UNSIGNED_BYTE; break;
        case 4:         image_internal_format = GL_RGBA8;   image_format = GL_BGRA; image_value_type = GL_UNSIGNED_BYTE; break;
        }
      } break;
      case FIT_INT16:     image_internal_format = GL_RG16I;   image_format = GL_RG;   image_value_type = GL_UNSIGNED_SHORT;  break;
      case FIT_UINT16:    image_internal_format = GL_R16;     image_format = GL_RED;  image_value_type = GL_UNSIGNED_SHORT;  break;
      case FIT_RGB16:     image_internal_format = GL_RGB16;   image_format = GL_RGB;  image_value_type = GL_UNSIGNED_SHORT;  break;
      case FIT_RGBA16:    image_internal_format = GL_RGBA16;  image_format = GL_RGBA; image_value_type = GL_UNSIGNED_SHORT;  break;
      case FIT_INT32:     image_internal_format = GL_R32I;    image_format = GL_RED;  image_value_type = GL_UNSIGNED_INT;    break;
      case FIT_UINT32:    image_internal_format = GL_R32UI;   image_format = GL_RED;  image_value_type = GL_UNSIGNED_INT;    break;
      case FIT_FLOAT:     image_internal_format = GL_R32F;    image_format = GL_RED;  image_value_type = GL_FLOAT;           break;
      case FIT_RGBF:      image_internal_format = GL_RGB32F;  image_format = GL_RGB;  image_value_type = GL_FLOAT;           break;
      case FIT_RGBAF:     image_internal_format = GL_RGBA32F; image_format = GL_RGBA; image_value_type = GL_FLOAT;           break;
      default: BOOST_LOG_TRIVIAL(error) << "texture2d::load(): " << " unknown image format in file: " << in_image_path << "Format : " << image_type << std::endl;       break;
      }

      glTexImage2D(t, 0, image_internal_format, image_width, image_height, 0, image_format, image_value_type, static_cast<void*>(in_image->accessPixels()));
    }


} } // namespace gpucast / namespace gl
