/********************************************************************************
* 
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : clutils.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_HUTILS_HPP
#define GPUCAST_GL_HUTILS_HPP

#include <cstdlib>
#include <iostream>
#include <string>

#if WIN32
  #pragma warning(disable: 4245)
#endif

#include <gpucast_gl/glpp.hpp>
#include <CL/cl.h>

namespace gpucast { namespace gl {

GPUCAST_GL std::string get_clerror (cl_int err);

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_HUTILS_HPP
