/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : glpp.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_HPP
#define GPUCAST_GL_HPP

// include glew before any other GL header
#include <GL/glew.h>

// disable glut/STL warnings
#ifdef WIN32
  #pragma warning(disable:4251) // dll-interface for stl container
  #pragma warning(disable:4275) // non dll-interface for boost::noncopyable
  #pragma warning(disable:4996) // std::copy unsafe
  #pragma warning(disable:4505) // glutInit_ATEXIT_HACK unreferenced local function has been removed
  #pragma warning(disable:4351) // default initialisation of array
#endif

#ifdef WIN32
  // disable openCL warnings
  #pragma warning(disable:4610) // can never be instantiated
  #pragma warning(disable:4512) // assignment operator could not be constructed
  #pragma warning(disable:4510) // deafult c'tor could not be constructed
  #pragma warning(disable:4482) // Qt::Mousebutton
  #pragma warning(disable:4100) // unreferenced parameter
#endif

#ifdef WIN32
  #if GPUCAST_GL_EXPORT
    #define GPUCAST_GL __declspec(dllexport)
  #else
    #define GPUCAST_GL __declspec(dllimport)
  #endif
#else
  #define GPUCAST_GL
#endif

#define GPUCAST_GL_DIRECT_STATE_ACCESS 1

#endif // GPUCAST_GL_HPP
