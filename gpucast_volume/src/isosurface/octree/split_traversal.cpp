/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : split_traversal.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/split_traversal.hpp"

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/split_criteria.hpp>
#include <gpucast/volume/isosurface/octree/ocnode.hpp>
#include <gpucast/volume/uid.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
split_traversal::split_traversal(split_criteria_ptr const& criteria )
  : _split_criteria ( criteria )
{}


///////////////////////////////////////////////////////////////////////////////
split_traversal::~split_traversal()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
split_traversal::visit ( ocnode& node ) const
{
  // first determine if node contains outerfaces
  auto face = node.face_begin();
  while ( face != node.face_end() )
  {
    if ( (**face).outer )
    {
      node.contains_outer_face ( true );
      break; 
    } else {
      ++face;
    }
  }

  // subdivide node
  if ( split_necessary ( node ) )
  {
    // octreemize bounding box
    std::vector<node::boundingbox_type> ocsplit;
    node.boundingbox().uniform_split ( std::back_inserter(ocsplit) );

    std::size_t ocsplit_id = 0;

    // create children nodes and apply their size
    for ( node::boundingbox_type const& bbox : ocsplit)
    {
      // apply axis aligned bounding box to new child node and add as child
      ocnode_ptr child_ptr = std::make_shared<ocnode> ( node.shared_from_this(), node.depth() + 1, bbox, uid::generate("ocnode") );
      node.add_node(child_ptr);
      
      bool init_range = true;

      for ( auto f = node.face_begin(); f != node.face_end(); ++f )
      {
        if ( (*f)->obb.overlap ( bbox ) )
        {
          child_ptr->add_face(*f);
          if ( init_range ) {
            child_ptr->range ( (*f)->attribute_range );
            init_range = false;
          } else {
            child_ptr->range (child_ptr->range () + (**f).attribute_range);
          }
        }
      }
    }
    node.clear_faces();
  }

  std::for_each(node.begin(), node.end(), std::bind(&node::accept, std::placeholders::_1, *this));
}

///////////////////////////////////////////////////////////////////////////////
bool              
split_traversal::split_necessary ( ocnode& node ) const
{
  return (*_split_criteria)(node);
}
} // namespace gpucast
