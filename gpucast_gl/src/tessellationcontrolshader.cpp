/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : tessellationcontrolshader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/tessellationcontrolshader.hpp"

// header system
#include <GL/glew.h>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
tessellationcontrolshader::tessellationcontrolshader()
  : shader()
{
  id_ = glCreateShader( GL_TESS_CONTROL_SHADER);
}


////////////////////////////////////////////////////////////////////////////////
tessellationcontrolshader::~tessellationcontrolshader()
{}

} } // namespace gpucast / namespace gl
