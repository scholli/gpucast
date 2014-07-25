/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : node.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_NODE_HPP
#define GPUCAST_NODE_HPP

// header, system
#include <list>

// header, external
#include <boost/array.hpp>
#include <memory>
#include <boost/unordered_set.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>

#include <gpucast/gl/math/matrix4x4.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/beziervolume.hpp>
#include <gpucast/volume/isosurface/face.hpp>

namespace gpucast {

class nodevisitor;
class node;
typedef std::shared_ptr<node> node_ptr;

class GPUCAST_VOLUME node : public std::enable_shared_from_this<node>

{
public : // typedefs / enums

  typedef beziervolume::value_type                      value_type;

  typedef gpucast::math::point<value_type,3>                      point_type;
  typedef gpucast::math::axis_aligned_boundingbox<point_type>     boundingbox_type;
  typedef std::shared_ptr<boundingbox_type>           boundingbox_ptr;

  typedef std::shared_ptr<node>                       pointer;
  typedef std::vector<node_ptr>                         nodecontainer;

  typedef nodecontainer::iterator                       iterator;
  typedef nodecontainer::const_iterator                 const_iterator;


public : // c'tor / d'tor

  node        ();

  node        ( pointer const&  parent,
                std::size_t     depth,
                std::size_t     id = 0 );

  ~node       ();

public : // methods

  virtual boundingbox_type const& boundingbox     () const = 0;

  // general information
  bool                          has_children      () const;
  bool                          has_parent        () const;

  std::size_t                   children          () const;
  std::vector<node_ptr> const&  get_children      () const;

  void                          clear             ();
  void                          add_node          ( node_ptr const& );

  void                          depth             ( std::size_t d );
  std::size_t                   depth             () const;

  virtual value_type            volume            () const = 0;
  virtual value_type            surface           () const = 0;

  // traverse/add/remove/children
  iterator                      begin             ();
  iterator                      end               ();

  const_iterator                begin             () const;
  const_iterator                end               () const;

  pointer const&                parent            ( ) const;
  void                          parent            ( pointer const& );

  void                                  id            ( std::size_t );
  std::size_t                           id            () const;

  virtual void                  accept            ( nodevisitor const& visitor ) = 0;

  virtual void                  compute_bbox_from_children () = 0;
  virtual void                  compute_bbox_from_data     () = 0;

  virtual void                  print             ( std::ostream& os ) const;

private : // auxilliary methods

private : // attributes

  pointer                       _parent;
  nodecontainer                 _children;
  std::size_t                   _depth;
  std::size_t                   _id;

};

std::ostream& operator<<(std::ostream& os, node const& node);

} // namespace gpucast

#endif // GPUCAST_NODE_HPP