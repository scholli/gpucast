/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_translation.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/test/dynamic_translation.hpp"

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  dynamic_translation::dynamic_translation( unsigned      dsteps,
                                            gpucast::math::vec3f const&  source_position,
                                            gpucast::math::vec3f const&  target_position )
    : dynamic_transform ( dsteps ),
      _source           ( source_position ),
      _target           ( target_position )
  {}


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_translation::~dynamic_translation()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ gpucast::math::matrix4f    
  dynamic_translation::current_transform()
  {
    float alpha = float(_current_step) / float(_discrete_steps);
    gpucast::math::vec3f t = (1.0f - alpha) * _source + alpha * _target;
    return gpucast::math::make_scale ( t[0], t[1], t[2] );
  }

} } // namespace gpucast / namespace gl

