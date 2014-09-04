/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_transform.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/test/dynamic_transform.hpp"

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  dynamic_transform::dynamic_transform( unsigned dsteps )
    : _discrete_steps ( dsteps ),
      _current_step   ( 0 )
  {}


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_transform::~dynamic_transform()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  void 
  dynamic_transform::reset()
  {
    _current_step = 0;
  }


  ///////////////////////////////////////////////////////////////////////////////
  bool 
  dynamic_transform::finished() const
  {
    return _current_step >= _discrete_steps;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                
  dynamic_transform::step ()
  {
    ++_current_step;
  }
   
} } // namespace gpucast / namespace gl

