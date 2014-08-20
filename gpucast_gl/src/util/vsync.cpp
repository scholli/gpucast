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
#else
  #include <GL/glxew.h>
  #include <GL/glx.h>
  #include <GL/glxext.h>
#endif


namespace gpucast { namespace gl {

bool set_vsync(bool enable)
{
#if WIN32
  if ( !wglSwapIntervalEXT ) 
  {
    return false;
  }
  wglSwapIntervalEXT(enable);
#else
  if( !glewGetExtension("GLX_EXT_swap_control") )
  {
    return false;
  }
  glXSwapIntervalEXT(glXGetCurrentDisplay(), glXGetCurrentDrawable(), int(enable));
#endif
  return true;
}


bool get_vsync(bool& vsync)
{
#if WIN32
  if( !wglGetSwapIntervalEXT )
  {
    return false;
  }
  
  vsync = 0 != wglGetSwapIntervalEXT();
#else
  if( !glewGetExtension("GLX_EXT_swap_control") )
  {
    return false;
  }
  unsigned int swap;
  glXQueryDrawable(glXGetCurrentDisplay(), glXGetCurrentDrawable(), GLX_SWAP_INTERVAL_EXT, &swap);
  vsync = 0 != swap; 
#endif
  return true;
}




} } // namespace gpucast / namespace gl

