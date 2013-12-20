/********************************************************************************
* 
* Copyright (C) 2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : tessellationevaluationshader.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_TESSELLATIONEVALUATIONSHADER_HPP
#define GPUCAST_GL_TESSELLATIONEVALUATIONSHADER_HPP

// header, project
#include <gpucast/gl/shader.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL tessellationevaluationshader : public shader
{
public :
  tessellationevaluationshader();
  virtual ~tessellationevaluationshader();
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_TESSELLATIONEVALUATIONSHADER_HPP
