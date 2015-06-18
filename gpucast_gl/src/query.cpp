/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vertexshader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/vertexshader.hpp"

// header system
#include <GL/glew.h>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
vertexshader::vertexshader()
  : shader()
{
  id_ = glCreateShader(GL_VERTEX_SHADER);
}


////////////////////////////////////////////////////////////////////////////////
vertexshader::~vertexshader()
{}

} } // namespace gpucast / namespace gl
