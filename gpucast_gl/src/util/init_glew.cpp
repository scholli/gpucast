/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : init_glew.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/util/init_glew.hpp"

namespace gpucast { namespace gl {

/////////////////////////////////////////////////////////////////////////////
void init_glew ( std::ostream& os )
{
  glewExperimental = true;
  GLenum err = glewInit();

  if (GLEW_OK != err) {
    os << "glewInit() failed" << std::endl;
    throw std::runtime_error("Could not initialize glew. GL Context ready?");
  } else {
    os << "glewInit() ok" << std::endl;

    // clear INVALID_ENUM error from stack
    err = glGetError();
  }
}

} } // namespace gpucast / namespace gl

