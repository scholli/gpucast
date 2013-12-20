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
                                    vec3f const& originalsize,
                                    vec3f const& targetsize )
    : dynamic_transform ( dsteps ),
      _originalsize     ( originalsize ),
      _targetsize       ( targetsize )
  {}


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_scaling::~dynamic_scaling()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  /* virtual */ matrix4f    
  dynamic_scaling::current_transform()
  {
    float alpha = float(_current_step) / float(_discrete_steps);
    vec3f s = (1.0f - alpha) * _originalsize + alpha * _targetsize;
    return make_scale ( s[0], s[1], s[2] );
  }

} } // namespace gpucast / namespace gl

