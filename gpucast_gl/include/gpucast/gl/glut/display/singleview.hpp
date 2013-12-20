/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : displaysetup.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_SINGLEVIEW_HPP
#define GPUCAST_GL_SINGLEVIEW_HPP

// header, system
#include <string>

#include <boost/function.hpp>

#include <gpucast/gl/framebufferobject.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/renderbuffer.hpp>
#include <gpucast/gl/math/vec3.hpp>

#include <gpucast/gl/glut/display/displaysetup.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL singleview : public displaysetup 
{
public :

  singleview                  ( unsigned      width, 
                                unsigned      height, 
                                vec3f const&  camera_position, 
                                vec3f const&  screen_position,
                                float         screenwidth,
                                float         screenheight );

  virtual         ~singleview ( );

  virtual void    display     ( );
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_SINGLEVIEW_HPP
