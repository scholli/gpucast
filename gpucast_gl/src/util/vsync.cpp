/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vsync.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/util/vsync.hpp"

// header, system
#include <iostream>

#if WIN32
  #include <GL/wglew.h>
#endif


namespace gpucast { namespace gl {

#if WIN32

bool set_vsync(bool enable)
{
  if ( !wglSwapIntervalEXT ) 
  {
    return false;
  }

  wglSwapIntervalEXT(enable);
  return true;
}


bool get_vsync(bool& vsync)
{
  if( !wglGetSwapIntervalEXT )
  {
    return false;
  }
  
  vsync = 0 != wglGetSwapIntervalEXT();
  return true;
}

#endif


} } // namespace gpucast / namespace gl

