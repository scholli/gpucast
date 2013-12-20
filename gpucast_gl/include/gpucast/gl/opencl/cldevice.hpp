/********************************************************************************
* 
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : cldevice.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CLDEVICE_HPP
#define GPUCAST_GL_CLDEVICE_HPP

#include <cstdlib>
#include <iostream>
#include <string>

#include <gpucast_gl/glpp.hpp>
#include <boost/shared_ptr.hpp>

// opencl header
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

namespace gpucast { namespace gl {

  class clcontext;

  class GPUCAST_GL cldevice
  {
    public : 

      friend class clplatform;

    private : // c'tor / d'tor

      cldevice    ();

    public :

      ~cldevice   ();

    public : // methods
            

      boost::shared_ptr<clcontext>  create_context  () const;

      void          print           ( std::ostream& os) const;

    private : // attributes

      cl_device_id     _id;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CLDEVICE_HPP
