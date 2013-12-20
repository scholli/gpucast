/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : clplatform.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast_gl/opencl/clplatform.hpp"

// system
#include <gpucast_gl/opencl/cldevice.hpp> 

namespace gpucast { namespace gl {

clplatform::clplatform(std::string const& name)
  : _id (0)
{
  platform_map platforms = get_available_platforms();

  for (platform_map::const_iterator i = platforms.begin(); i != platforms.end(); ++i) 
  {
    if (i->second.find_first_of(name, 0) < i->second.size()) {
      _id = i->first;
      break;
    }
  }

  // default to zeroeth platform if search string was not found
  if (_id == 0)
  {
    std::cerr << "Requested OpenCL platform not found - default set to zeroeth platform.\n" << std::endl;
    _id = platforms.begin()->first;
  }
}


///////////////////////////////////////////////////////////////////////////////
clplatform::~clplatform()
{}


///////////////////////////////////////////////////////////////////////////////
clplatform::device_ptr            
clplatform::get_device ( cl_device_type type ) const
{
  device_ptr device ( new cldevice );
  clGetDeviceIDs    ( _id, type, 1, &device->_id, 0);
  return device;
}


///////////////////////////////////////////////////////////////////////////////
/*static*/ clplatform::platform_map
clplatform::get_available_platforms ()
{
  platform_map platforms;

  char              buffer[1024];
  cl_uint           num_platforms; 
  cl_platform_id*   platform_ids;
  cl_int            err;

  // request number of platforms
  err = clGetPlatformIDs ( 0, 0, &num_platforms );

  // return if there are no platforms available 
  if ( err != CL_SUCCESS || num_platforms == 0 ) {
    throw std::runtime_error("clGetPlatformIDs() failed or no OpenCL platform available.");
  }

  // allocate memory for platform ids
  platform_ids = new cl_platform_id[num_platforms * sizeof(cl_platform_id)];

  // request platform ids
  err = clGetPlatformIDs (num_platforms, platform_ids, 0);

  // iterate all available platforms
  for ( cl_uint i = 0; i < num_platforms; ++i )
  {
    // request platform name
    err = clGetPlatformInfo ( platform_ids[i], CL_PLATFORM_NAME, 1024, &buffer, 0 );

    if ( err == CL_SUCCESS )
    {
      platforms.insert ( std::make_pair(platform_ids[i], buffer) );
    } else {
      throw std::runtime_error("Error requesting OpenCL platform name.");
    }
  }

  delete[] platform_ids;

  return platforms;
}

} } // namespace gpucast / namespace gl
