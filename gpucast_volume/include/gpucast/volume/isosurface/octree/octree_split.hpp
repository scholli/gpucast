/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : octree_split.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_OCTREE_SPLIT_HPP
#define GPUCAST_OCTREE_SPLIT_HPP

// header, system
#include <memory>

// header, project
#include <gpucast/volume/isosurface/octree/split_traversal.hpp>
#include <gpucast/volume/isosurface/octree/split_criteria.hpp>

namespace gpucast {

// forward declaration
class ocnode;

class octree_split : public split_traversal
{
  public :

    octree_split::octree_split(split_criteria_ptr const& criteria);

    node_ptr create_node() const;

    /*virtual*/ void visit(ocnode& node) const override;

private : 

};

} // namespace gpucast

#endif // GPUCAST_OCTREE_SPLIT_HPP
