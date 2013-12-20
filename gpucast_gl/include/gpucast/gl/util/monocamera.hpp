/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : monocamera.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_MONO_CAMERA_HPP
#define GPUCAST_GL_MONO_CAMERA_HPP

// header, system
#include <string>

#include <boost/function.hpp>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/camera.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL monocamera : public camera 
{
public :

  monocamera() {};

  virtual         ~monocamera() {};

  virtual void    display() {};
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_MONO_CAMERA_HPP
