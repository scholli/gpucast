/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : fragmentshader.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_FRAGMENTSHADER_HPP
#define GPUCAST_GL_FRAGMENTSHADER_HPP

#include <gpucast/gl/shader.hpp>

namespace gpucast { namespace gl {

class GPUCAST_GL fragmentshader : public shader
{
public :
  fragmentshader();
  virtual ~fragmentshader();
} ;

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_FRAGMENTSHADER_HPP


