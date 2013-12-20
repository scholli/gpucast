/********************************************************************************
* 
* Copyright (C) 2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : tessellationcontrolshader.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_TESSELLATIONCONTROLSHADER_HPP
#define GPUCAST_GL_TESSELLATIONCONTROLSHADER_HPP

// header, project
#include <gpucast/gl/shader.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL tessellationcontrolshader : public shader
{
public :
  tessellationcontrolshader();
  virtual ~tessellationcontrolshader();
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_TESSELLATIONCONTROLSHADER_HPP
