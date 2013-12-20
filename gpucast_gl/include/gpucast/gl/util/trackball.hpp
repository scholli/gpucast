/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trackball.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef TRACKBALL_HPP
#define TRACKBALL_HPP

// header, system

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/eventhandler.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>



namespace gpucast { namespace gl {

class GPUCAST_GL trackball : public eventhandler
{
public :

public :

  trackball                         ( float mappingzoom_   = 1.0f,
                                      float mappingshift_  = 0.5f,
                                      float mappingrotate_ = 0.25f );

  /* virtual */ ~trackball          ( );

public :

  matrix4x4<float>        rotation  ( ) const;

  float                   distance  ( ) const;
  float                   shiftx    ( ) const;
  float                   shifty    ( ) const;

  void                    reset     ( );

  /* virtual */ void      mouse    ( enum button, enum state, int x, int y );
  /* virtual */ void      motion   ( int x, int y );
  /* virtual */ void      passivemotion ( int x, int y );
  /* virtual */ void      keyboard ( unsigned char, int, int);

private :

  // current button state
  bool                    button_left_;
  bool                    button_middle_;
  bool                    button_right_;

  // current pixel position
  int                     mousepos_x_;
  int                     mousepos_y_;

  // current rotation
  matrix4x4<float>        rotation_euler_;

  // current distance
  float                   distance_;

  // current shift
  float                   shiftx_;
  float                   shifty_;

  // mapping from mouse motion to movement
  struct {
    float mapping_zoom;      // zoom factor, default 1.0
    float mapping_shift;     // shift factor, default 0.5
    float mapping_rotate;    // degree per pixel, default 0.25
  }                       config_;
};

} } // namespace gpucast / namespace gl

#endif // TRACKBALL_HPP
