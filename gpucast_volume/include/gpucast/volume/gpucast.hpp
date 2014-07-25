/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : gpucast.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_HPP
#define GPUCAST_VOLUME_HPP

#if 0
  #define _CRTDBG_MAP_ALLOC
  #include <stdlib.h>
  #include <crtdbg.h>
#endif

#if WIN32
  #pragma warning(disable:4251) // dll-interface for stl container
  #pragma warning(disable:4275) // dll-interface for stl container
  #pragma warning(disable:4996) // std::copy unsafe
#endif

#if WIN32
  #if GPUCAST_VOLUME_EXPORT
    #define GPUCAST_VOLUME __declspec(dllexport)
    #define GPUCAST_VOLUME_EXTERN
  #else
    #define GPUCAST_VOLUME __declspec(dllimport)
    #define GPUCAST_VOLUME_EXTERN extern
  #endif
#else
  #define GPUCAST_VOLUME
  #define GPUCAST_VOLUME_EXTERN
#endif

#include <GL/glew.h>

#endif // GPUCAST_HPP
