/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : geometryshader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/geometryshader.hpp"

// header system
#include <GL/glew.h>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
geometryshader::geometryshader()
  : shader()
{
  id_ = glCreateShader(GL_GEOMETRY_SHADER);
}


////////////////////////////////////////////////////////////////////////////////
geometryshader::~geometryshader()
{}

} } // namespace gpucast / namespace gl
