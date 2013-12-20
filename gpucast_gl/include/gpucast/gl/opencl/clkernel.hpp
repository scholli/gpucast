/********************************************************************************
* 
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : clkernel.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CLKERNEL_HPP
#define GPUCAST_GL_CLKERNEL_HPP

#include <cstdlib>
#include <iostream>
#include <string>

#include <gpucast_gl/glpp.hpp>

// opencl header
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

namespace gpucast { namespace gl {

  class GPUCAST_GL clkernel 
  {
    public : // c'tor / d'tor

      clkernel();
      ~clkernel();

    public : // methods

    private : // attributes

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CLKERNEL_HPP
