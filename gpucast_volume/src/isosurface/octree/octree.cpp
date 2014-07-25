/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : octree.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/octree.hpp"

// header, system

// header, external
#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>

// header, project
#include <gpucast/volume/uid.hpp>
#include <gpucast/volume/volume_renderer.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/volume/isosurface/octree/info_traversal.hpp>
#include <gpucast/volume/isosurface/face.hpp>
#include <gpucast/volume/isosurface/octree/ocnode.hpp>
#include <gpucast/volume/isosurface/octree/split_traversal.hpp>

namespace gpucast {


///////////////////////////////////////////////////////////////////////////////
octree::octree()
{}


///////////////////////////////////////////////////////////////////////////////
octree::~octree()
{}


///////////////////////////////////////////////////////////////////////////////
void
octree::generate  ( volume_ptr const&             volume,
                    std::vector<face_ptr> const&  faces,
                    split_traversal const&        splitter,
                    volume_type::boundingbox_type::value_type const& border_offset_percentage
                    )
{
  // store pointer to object octree is built for
  _object = volume;

  // border set bbox
  _boundingbox.min = volume->bbox().min - border_offset_percentage * volume->bbox().size();
  _boundingbox.max = volume->bbox().max + border_offset_percentage * volume->bbox().size();

  // create new root-node
  _root.reset( new ocnode ( std::shared_ptr<node>(), 0, uid::generate("ocnode") ) );

  // add all found faces to root node
  std::for_each ( faces.begin(), faces.end(), std::bind(&ocnode::add_face, std::ref(_root), std::placeholders::_1) );

  // apply volume's geometric bounding box to node
  _root->boundingbox ( _boundingbox );

  // apply volume attribute range to root node
  std::string attribute_name = *(volume->parent()->attribute_list().begin());
  nurbsvolumeobject::attribute_boundingbox attribute_range = volume->parent()->bbox ( attribute_name ); 
  _root->range ( ocnode::interval_t ( attribute_range.min[0], attribute_range.max[0] ) );

  // start splitting process
  _root->accept ( splitter );
}


///////////////////////////////////////////////////////////////////////////////
void            
octree::accept ( nodevisitor& v)
{
  _root->accept(v);
}


///////////////////////////////////////////////////////////////////////////////
ocnode_ptr const&   
octree::root () const
{
  return _root;
}


///////////////////////////////////////////////////////////////////////////////
node::point_type
octree::min () const
{
  return _boundingbox.min;
}

///////////////////////////////////////////////////////////////////////////////
node::point_type
octree::max () const
{
  return _boundingbox.max;
}

} // namespace gpucast

