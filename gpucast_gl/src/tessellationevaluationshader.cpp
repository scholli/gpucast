/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : tessellationevaluationshader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/tessellationevaluationshader.hpp"

// header system
#include <GL/glew.h>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
tessellationevaluationshader::tessellationevaluationshader()
  : shader()
{
  id_ = glCreateShader(GL_TESS_EVALUATION_SHADER);
}


////////////////////////////////////////////////////////////////////////////////
tessellationevaluationshader::~tessellationevaluationshader()
{}

} } // namespace gpucast / namespace gl
