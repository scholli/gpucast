/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : error.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// header i/f
#include "gpucast/gl/error.hpp"

// header, system
#include <iostream>
#include <boost/log/trivial.hpp>

#include <GL/glew.h>

namespace gpucast { namespace gl {

bool
error(std::string const& message)
{
  GLenum err = glGetError();

  BOOST_LOG_TRIVIAL(error) << message << " : ";

  switch (err) {

    case GL_NO_ERROR :
      BOOST_LOG_TRIVIAL(error) << "no error.\n";
      return true;

    case GL_INVALID_ENUM :
      BOOST_LOG_TRIVIAL(error) << "GL_INVALID_ENUM" << std::endl;
      return false;

    case GL_INVALID_VALUE :
      BOOST_LOG_TRIVIAL(error) << "GL_INVALID_VALUE" << std::endl;
      return false;

    case GL_INVALID_OPERATION :
      BOOST_LOG_TRIVIAL(error) << "GL_INVALID_OPERATION" << std::endl;
      return false;

    case GL_STACK_OVERFLOW :
      BOOST_LOG_TRIVIAL(error) << "GL_STACK_OVERFLOW" << std::endl;
      return false;

    case GL_STACK_UNDERFLOW :
      BOOST_LOG_TRIVIAL(error) << "GL_STACK_UNDERFLOW" << std::endl;
      return false;

    case GL_OUT_OF_MEMORY :
      BOOST_LOG_TRIVIAL(error) << "GL_OUT_OF_MEMORY" << std::endl;
      return false;

    case GL_TABLE_TOO_LARGE :
      BOOST_LOG_TRIVIAL(error) << "GL_TABLE_TOO_LARGE" << std::endl;
      return false;

    case GL_INVALID_FRAMEBUFFER_OPERATION_EXT :
      BOOST_LOG_TRIVIAL(error) << "GL_INVALID_FRAMEBUFFER_OPERATION_EXT" << std::endl;
      return false;

    default :
      BOOST_LOG_TRIVIAL(error) << "unknwon error.\n";
      return false;
  };
}

} } // namespace gpucast / namespace gl
