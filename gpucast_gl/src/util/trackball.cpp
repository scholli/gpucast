/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trackball.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/gl/util/trackball.hpp"

// header, system
#include <cassert> // assert

#include <cmath>

// header, project

namespace gpucast { namespace gl {

  ////////////////////////////////////////////////////////////////////////////////
  trackball::trackball( float zoom,
                        float shift,
                        float rotation )
    :
      eventhandler    ( ),
      button_left_    ( false ),
      button_middle_  ( false ),
      button_right_   ( false ),
      mousepos_x_     ( 0 ),
      mousepos_y_     ( 0 ),
      rotation_euler_ (   ),
      distance_       ( 0.0 ),
      shiftx_         ( 0.0 ),
      shifty_         ( 0.0 ),
      config_         (   )
  {
    config_.mapping_zoom    = zoom;
    config_.mapping_shift   = shift;
    config_.mapping_rotate  = rotation;
  }



  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ trackball::~trackball( )
  {}


  ////////////////////////////////////////////////////////////////////////////////
  matrix4x4<float>
  trackball::rotation() const
  {
    return rotation_euler_;
  }

  ////////////////////////////////////////////////////////////////////////////////
  float
  trackball::distance() const
  {
    return distance_;
  }


  ////////////////////////////////////////////////////////////////////////////////
  float
  trackball::shiftx() const
  {
    return shiftx_;
  }


  ////////////////////////////////////////////////////////////////////////////////
  float
  trackball::shifty() const
  {
    return shifty_;
  }


  ////////////////////////////////////////////////////////////////////////////////
  void                    
  trackball::reset ( )
  {
    button_left_     = false;
    button_middle_   = false;
    button_right_    = false;
    mousepos_x_      = 0;
    mousepos_y_      = 0;
    rotation_euler_  = matrix4f();
    distance_        = 0.0; 
    shiftx_          = 0.0;
    shifty_          = 0.0;
  }

  ////////////////////////////////////////////////////////////////////////////////
  int trackball::posx() const
  {
    return mousepos_x_;
  }

  ////////////////////////////////////////////////////////////////////////////////
  int trackball::posy() const
  {
    return mousepos_y_;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  trackball::mouse(enum button b, enum state s, int x, int y)
  {
    mousepos_x_ = x;
    mousepos_y_ = y;

    if ( b == left )
    {
      if ( s == press )
      {
        button_left_ = true;
      } else {
        if ( s == release )
        {
          button_left_ = false;
        }
      }
    }

    if ( b == right )
    {
      if ( s == press )
      {
        button_right_ = true;
      } else {
        if ( s == release )
        {
          button_right_ = false;
        }
      }
    }

    if(b == middle )
    {
      if(s == press )
      {
        button_middle_ = true;
      } else {
        if (s == release )
        {
          button_middle_ = false;
        }
      }
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  trackball::motion(int x, int y)
  {
    if ( button_left_ )
    {
      float rady = (config_.mapping_rotate * 2.0f * float(M_PI) * float(x - mousepos_x_)) / 360.0f;
      float radx = (config_.mapping_rotate * 2.0f * float(M_PI) * float(y - mousepos_y_)) / 360.0f;

      rotation_euler_ = make_rotation_x(-radx) * rotation_euler_;
      rotation_euler_ = make_rotation_y(-rady) * rotation_euler_;

      mousepos_x_ = x;
      mousepos_y_ = y;
    }

    if ( button_right_ )
    {
      distance_ += config_.mapping_zoom * float(y - mousepos_y_);
      mousepos_y_ = y;
    }

    if ( button_middle_ )
    {
      shiftx_ += config_.mapping_shift * float(x - mousepos_x_);
      shifty_ -= config_.mapping_shift * float(y - mousepos_y_);
      mousepos_y_ = y;
      mousepos_x_ = x;
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  trackball::passivemotion ( int, int )
  {
    // no use in trackball
  }


  ////////////////////////////////////////////////////////////////////////////////
  /* virtual */ void
  trackball::keyboard ( unsigned char, int, int)
  {
    // no use in trackball
  }

  } } // namespace gpucast / namespace gl
