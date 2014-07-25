/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : draw_traversal.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/draw_traversal.hpp"

// header, system
#include <functional>

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>



namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
draw_traversal::draw_traversal(gpucast::gl::matrix4x4<float> const& mvp)
  : _mvp(mvp)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
draw_traversal::visit ( ocnode& n ) const
{
  n.draw(_mvp);
  std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, *this));
}

} // namespace gpucast
