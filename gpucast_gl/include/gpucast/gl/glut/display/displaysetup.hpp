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
#ifndef GPUCAST_GL_DISPLAYSETUP_HPP
#define GPUCAST_GL_DISPLAYSETUP_HPP

// header, system
#include <string>
#include <functional>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/math/vec3.hpp>

namespace gpucast { namespace gl {

class trackball;
class camera;

///////////////////////////////////////////////////////////////////////////////
// display setup base class
///////////////////////////////////////////////////////////////////////////////
class GPUCAST_GL displaysetup
{
public :

  displaysetup                  (std::string const&             name,
                                 unsigned                       width,
                                 unsigned                       height,
                                 float                          screenwidth,
                                 float                          screenheight,
                                 vec3f const&                   camera_position,
                                 vec3f const&                   screen_position );

  virtual        ~displaysetup  ();

  virtual void   display        ( ) = 0;

  virtual void   resize         ( unsigned    width,
                                  unsigned    height);

  void           set_drawfun    ( std::function<void (void*)> draw_fun,
                                  void*                         userdata_ );

  void           set_camera     ( camera* cam );

  unsigned       width          ( ) const;

  unsigned       height         ( ) const;

  void           mouse          ( int button, int state, int x ,int y) const;
  void           motion         ( int x ,int y) const;

protected :

  virtual void   viewsetup      ( float       camera_x,
                                  float       camera_y,
                                  float       camera_z,
                                  float       target_x,
                                  float       target_y,
                                  float       target_z );
private :

  std::string                     _name;

  unsigned                        _width;
  unsigned                        _height;

protected :

  std::function<void (void*)>   _draw_fun;
  void*                           _userdata;

  float                           _screenwidth;
  float                           _screenheight;

  camera*                         _camera;
  trackball*                      _trackball;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DISPLAYSETUP_HPP
