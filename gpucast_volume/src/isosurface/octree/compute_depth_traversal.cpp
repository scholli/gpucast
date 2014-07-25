/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_depth_traversal.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/compute_depth_traversal.hpp"

// header, system

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
compute_depth_traversal::compute_depth_traversal(std::size_t start_depth)
: _current_depth(start_depth)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
compute_depth_traversal::visit ( ocnode& o ) const
{
  o.depth(_current_depth);
  std::for_each(o.begin(), o.end(), std::bind(&node::accept, std::placeholders::_1, compute_depth_traversal(_current_depth + 1)));
}

} // namespace gpucast
