/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : compute_depth_traversal.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_COMPUTE_DEPTH_TRAVERSAL_HPP
#define GPUCAST_COMPUTE_DEPTH_TRAVERSAL_HPP

// header, system

// header, project
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
class compute_depth_traversal : public nodevisitor
{
public :

  compute_depth_traversal ( std::size_t start_depth );
  /* virtual */ void      visit          ( ocnode& ) const;

private :

  std::size_t _current_depth;

};

} // namespace gpucast

#endif // GPUCAST_GENERIC_TREE_TRAVERSAL_HPP
