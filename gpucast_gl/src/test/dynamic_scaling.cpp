/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_scaling.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/test/dynamic_scaling.hpp"

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  dynamic_scaling::dynamic_scaling( unsigned dsteps,
                                    gpucast::math::vec3f const& originalsize,
                                    gpucast::math::vec3f const& targetsize )
    : dynamic_transform ( dsteps ),
      _originalsize     ( originalsize ),
      _targetsize       ( targetsize )
  {}


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_scaling::~dynamic_scaling()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ gpucast::math::matrix4f    
  dynamic_scaling::current_transform()
  {
    float alpha = float(_current_step) / float(_discrete_steps);
    gpucast::math::vec3f s = (1.0f - alpha) * _originalsize + alpha * _targetsize;
    return gpucast::math::make_scale ( s[0], s[1], s[2] );
  }

} } // namespace gpucast / namespace gl

