/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : split_by_volumecount.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/split_by_volumecount.hpp"

// header, system

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
split_by_volumecount::split_by_volumecount ( std::size_t max_depth, std::size_t max_volumes_per_node )
: split_criteria        (),
  _max_depth            ( max_depth ),
  _max_volumes_per_node ( max_volumes_per_node )
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ bool 
split_by_volumecount::operator() ( ocnode const& node ) const
{
  return node.faces() > _max_volumes_per_node && node.depth() < _max_depth;
}

} // namespace gpucast
