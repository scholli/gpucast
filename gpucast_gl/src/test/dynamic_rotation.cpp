/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_rotation.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/test/dynamic_rotation.hpp"

#include <cmath>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  dynamic_rotation::dynamic_rotation( unsigned dsteps,
                                      unsigned axis )
    : dynamic_transform ( dsteps ),
      _rotation_axis    ( axis )
  {}


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_rotation::~dynamic_rotation()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ matrix4f    
  dynamic_rotation::current_transform()
  {
    assert ( _rotation_axis <= 2 );

    float rotation_angle = (float(_current_step) / float(_discrete_steps)) * 2.0f * float(M_PI);

    switch ( _rotation_axis )
    {
    case 0 : 
      return make_rotation_x ( rotation_angle );
    case 1 : 
      return make_rotation_y ( rotation_angle );
    case 2 :
      return make_rotation_z ( rotation_angle );
    default :
      throw std::runtime_error("dynamic_rotation::step(): Invalid rotation axis.\n");
    };
  }

} } // namespace gpucast / namespace gl

