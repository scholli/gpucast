/********************************************************************************
* 
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : clprogram.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CLPROGRAM_HPP
#define GPUCAST_GL_CLPROGRAM_HPP

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

  class GPUCAST_GL clprogram 
  {
    public : // c'tor / d'tor

      clprogram();
      ~clprogram();

    public : // methods

    private : // attributes

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CLPROGRAM_HPP
