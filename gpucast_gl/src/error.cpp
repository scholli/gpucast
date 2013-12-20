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

#include <GL/glew.h>

namespace gpucast { namespace gl {

bool
error(std::string const& message)
{
  GLenum err = glGetError();

  std::cerr << message << " : ";

  switch (err) {

    case GL_NO_ERROR :
      std::cerr << "no error.\n";
      return true;

    case GL_INVALID_ENUM :
      std::cerr << "GL_INVALID_ENUM" << std::endl;
      return false;

    case GL_INVALID_VALUE :
      std::cerr << "GL_INVALID_VALUE" << std::endl;
      return false;

    case GL_INVALID_OPERATION :
      std::cerr << "GL_INVALID_OPERATION" << std::endl;
      return false;

    case GL_STACK_OVERFLOW :
      std::cerr << "GL_STACK_OVERFLOW" << std::endl;
      return false;

    case GL_STACK_UNDERFLOW :
      std::cerr << "GL_STACK_UNDERFLOW" << std::endl;
      return false;

    case GL_OUT_OF_MEMORY :
      std::cerr << "GL_OUT_OF_MEMORY" << std::endl;
      return false;

    case GL_TABLE_TOO_LARGE :
      std::cerr << "GL_TABLE_TOO_LARGE" << std::endl;
      return false;

    case GL_INVALID_FRAMEBUFFER_OPERATION_EXT :
      std::cerr << "GL_INVALID_FRAMEBUFFER_OPERATION_EXT" << std::endl;
      return false;

    default :
      std::cerr << "unknwon error.\n";
      return false;
  };
}

} } // namespace gpucast / namespace gl
