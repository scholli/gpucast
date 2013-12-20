/********************************************************************************
* 
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : clcontext.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CLCONTEXT_HPP
#define GPUCAST_GL_CLCONTEXT_HPP

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

  class GPUCAST_GL clcontext
  { 
    public : // friends, enums, typedefs

      friend class cldevice;

    private : // c'tor

      clcontext();
        
    public : // d'tor

      ~clcontext();

    public : // methods

    private : // attributes

      cl_context          _id;
      cl_command_queue    _queue;
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CLCONTEXT_HPP
