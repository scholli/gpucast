/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : fragmentshader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/fragmentshader.hpp"

// header, system

// header, project
#include <GL/glew.h>


namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
fragmentshader::fragmentshader()
  : shader()
{
  id_ = glCreateShader(GL_FRAGMENT_SHADER);
}

////////////////////////////////////////////////////////////////////////////////
fragmentshader::~fragmentshader()
{}


} } // namespace gpucast / namespace gl
