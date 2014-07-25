/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : octree_split.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/octree_split.hpp"

#include <memory>

// header, external
#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>
#include <gpucast/volume/beziervolume.hpp>
#include <gpucast/volume/isosurface/octree/split_by_volumecount.hpp>


namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
octree_split::octree_split(split_criteria_ptr const& criteria)
: split_traversal(criteria)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ octree_split::node_ptr
octree_split::create_node () const
{
  return node_ptr( new ocnode );
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
octree_split::visit ( ocnode& node ) const
{
  if ( !node.has_parent() ) // entering root node
  {
    // generate bounding box for root node
    ocnode::boundingbox_type root_bbox ( ocnode::point_type::maximum(), ocnode::point_type::minimum() );

    // therefore accumulate bounding boxes of all volume elements in node
    //for (auto i = node.subvolume_begin(); i != node.subvolume_end(); ++i)
    for (auto i = node.face_begin(); i != node.face_begin(); ++i)
    {
      //root_bbox.merge( i->volume.bbox() );  // use aabb -> internal obb's might overlap!
      gpucast::math::oriented_boundingbox<ocnode::point_type>     obb = (*i)->obb;// ox_by_mesh(gpucast::math::partial_derivative_policy<ocnode::point_type>(true));
      gpucast::math::axis_aligned_boundingbox<ocnode::point_type> aabb = obb.aabb();
      root_bbox.merge ( aabb );
    }

    // add a 2% offset around total bounding box
    root_bbox.scale(0.02f);

    // set bounding box
    node.boundingbox ( root_bbox );
    node.depth(0);
  }

  // subdivide node
  //if ( !node.empty() && node.depth() < _max_depth )
  if ( split_necessary(node) )
  {
    // octreemize bounding box
    std::list<node::boundingbox_ptr> ocsplit;
    node.boundingbox().uniform_split_ptr(std::back_inserter(ocsplit));

    std::size_t ocsplit_id = 0;

    // create children nodes and apply their size
    for(node::boundingbox_ptr const& abox : ocsplit)
    {
      std::shared_ptr<ocnode::boundingbox_type> b = std::dynamic_pointer_cast<ocnode::boundingbox_type>(abox);

      // apply axis aligned bounding box to new child node and add as child
      std::shared_ptr<gpucast::ocnode> child_ptr = std::make_shared<ocnode>(node.shared_from_this(), node.depth() + 1, *b, ocsplit_id++ );
      node.add_node(child_ptr);

      for (auto i = node.face_begin(); i != node.face_begin(); ++i)
      {
        // if obb of subvolume overlaps childnode's bbox add subvolume to child
        //if ( i->volume.obbox_by_mesh(gpucast::math::greedy_policy<node::point_type>()).overlap(*b) )
        gpucast::math::oriented_boundingbox<ocnode::point_type> ocnode_obb = *b;
        gpucast::math::oriented_boundingbox<ocnode::point_type> volume_obb = (*i)->obb;// volume.obbox_by_mesh(gpucast::math::partial_derivative_policy<node::point_type>(true));

        //if ( volume_obb.overlap(ocnode_obb) )
        if ( ocnode_obb.overlap(volume_obb)  )
        //if ( i->volume.obbox_by_mesh(gpucast::math::partial_derivative_policy<node::point_type>(true)).overlap(*b) )
        //if ( i->volume.obbox_by_mesh<gpucast::math::random_policy>().overlap(b) )
        //if ( i->volume.bbox().overlap(b) )
        {
          child_ptr->add_face(*i);
        }
      }
    }
    //node.remove_all_data();
  }

  std::for_each(node.begin(), node.end(), std::bind(&node::accept, std::placeholders::_1, *this));
}

} // namespace gpucast
