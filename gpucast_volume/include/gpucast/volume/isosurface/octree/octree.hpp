/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : octree.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_OCTREE_HPP
#define GPUCAST_OCTREE_HPP

// header, system
#include <unordered_set>
#include <memory>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/isosurface/face.hpp>
#include <gpucast/volume/isosurface/octree/ocnode.hpp>


namespace gpucast {

class split_traversal;
class nodevisitor;

class octree 
{
  public : // typedefs / enums

    typedef beziervolumeobject                      volume_type;
    typedef std::shared_ptr<volume_type>          volume_ptr;
    typedef std::unordered_set<volume_ptr>        volume_set;

    typedef std::shared_ptr<split_traversal>      split_ptr;

  public : // c'tor / d'tor

    octree  ();
    ~octree ();

  public : // methods

     // fill tree and build it
    void                      generate ( volume_ptr const&                                volume,
                                         std::vector<face_ptr> const&                     faces,
                                         split_traversal const&                           split_traversal,
                                         volume_type::boundingbox_type::value_type const& border_offset_percentage = 0.0 );

    // visit tree nodes
    void                      accept ( nodevisitor& );

    node::point_type          min () const;
    node::point_type          max () const;

    // get root node
    ocnode_ptr const&         root () const; 

  private : // auxilliary methods

  private : // attributes
    
    volume_ptr                      _object;
    volume_type::boundingbox_type   _boundingbox;
    ocnode_ptr                      _root;
};

} // namespace gpucast

#endif // GPUCAST_OCTREE_HPP