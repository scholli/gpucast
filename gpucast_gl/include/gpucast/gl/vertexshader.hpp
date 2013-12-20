/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : vertexshader.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_VERTEXSHADER_HPP
#define GPUCAST_GL_VERTEXSHADER_HPP

// header, project
#include <gpucast/gl/shader.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL vertexshader : public shader
{
public :
  vertexshader();
  virtual ~vertexshader();
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_VERTEXSHADER_HPP
