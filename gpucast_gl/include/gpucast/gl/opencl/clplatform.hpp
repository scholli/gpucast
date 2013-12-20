/********************************************************************************
* 
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : clplatform.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CLPLATFORM_HPP
#define GPUCAST_GL_CLPLATFORM_HPP

#include <map>
#include <iostream>
#include <string>

#include <gpucast_gl/glpp.hpp>
#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>

// opencl header
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

namespace gpucast { namespace gl {

  class cldevice;

  class GPUCAST_GL clplatform
  {
    public : // typedefs

      typedef boost::unordered_map<cl_platform_id, std::string>          platform_map;
      typedef boost::shared_ptr<cldevice>                                device_ptr;

    public : // c'tor / d'tor

      clplatform  ( std::string const& search_name );
      ~clplatform ();

    public : // methods

      static platform_map   get_available_platforms ();
  
      device_ptr            get_device              ( cl_device_type type = CL_DEVICE_TYPE_GPU ) const;

    private : // attributes

      cl_platform_id      _id;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CLPLATFORM_HPP
