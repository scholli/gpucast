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
#ifndef GPUCAST_CORE_HPP
#define GPUCAST_CORE_HPP

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
  #if GPUCAST_CORE_EXPORT
    #define GPUCAST_CORE __declspec(dllexport)
    #define GPUCAST_CORE_EXTERN
  #else
    #define GPUCAST_CORE __declspec(dllimport)
    #define GPUCAST_CORE_EXTERN extern
  #endif
#else
  #define GPUCAST_CORE
  #define GPUCAST_CORE_EXTERN
#endif

#endif // GPUCAST_CORE_HPP
