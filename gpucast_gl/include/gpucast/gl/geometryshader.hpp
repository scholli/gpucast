/********************************************************************************
* 
* Copyright (C) 2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : geometryshader.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_GEOMETRYSHADER_HPP
#define GPUCAST_GL_GEOMETRYSHADER_HPP

// header, project
#include <gpucast/gl/shader.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL geometryshader : public shader
{
public :
  geometryshader();
  virtual ~geometryshader();
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_GEOMETRYSHADER_HPP
