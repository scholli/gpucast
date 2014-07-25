/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : split_traversal.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_SPLIT_TRAVERSAL_HPP
#define GPUCAST_SPLIT_TRAVERSAL_HPP

// header, system
#include <memory>

// header, project
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>
#include <gpucast/volume/isosurface/octree/node.hpp>

namespace gpucast {

class split_criteria;
class ocnode;

typedef std::shared_ptr<split_criteria> split_criteria_ptr;
typedef std::shared_ptr<node>           node_ptr;

///////////////////////////////////////////////////////////////////////////////
class split_traversal : public nodevisitor
{
  public : // c'tor / d'tor

    split_traversal           ( split_criteria_ptr const& );
    virtual ~split_traversal  ();

  public : // methods

    /* virtual */ void      visit               ( ocnode& ) const;

    bool                    split_necessary     ( ocnode& node ) const;

  private : // attributes

    split_criteria_ptr      _split_criteria;
};

} // namespace gpucast

#endif // GPUCAST_SPLIT_TRAVERSAL_HPP
