/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : generic_tree_traversal.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
//#include "gpucast/volume/isosurface/octree/octree_traversal.hpp"

// header, system
#include <iostream>
#include <map>
#include <string>

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>



namespace gpucast {

#if 0
///////////////////////////////////////////////////////////////////////////////
ocsplit_by_volumecount::ocsplit_by_volumecount ( std::size_t max_depth, std::size_t max_volumes_per_node )
: _max_depth(max_depth),
  _max_volumes_per_node(max_volumes_per_node)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ bool 
ocsplit_by_volumecount::operator() ( ocnode const& node ) const
{
  return node.size() > _max_volumes_per_node && node.depth() < _max_depth;
}


///////////////////////////////////////////////////////////////////////////////
nodevisitor::~nodevisitor()
{}

///////////////////////////////////////////////////////////////////////////////
partition_traversal::~partition_traversal()
{}

///////////////////////////////////////////////////////////////////////////////
refinement_traversal::~refinement_traversal()
{}

///////////////////////////////////////////////////////////////////////////////
octree_partition::octree_partition(std::size_t max_depth)
: _max_depth(max_depth)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ octree_partition::node_ptr
octree_partition::create_root () const
{
  return node_ptr( new ocnode() );
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
octree_partition::visit ( ocnode& node )
{
  if ( !node.has_parent() ) // entering root node
  {
    // generate bounding box for root node
    ocnode::boundingbox_type root_bbox ( ocnode::point_type::maximum(), ocnode::point_type::minimum() );

    // therefore accumulate bounding boxes of all volume elements in node
    for (node::const_subvolume_iterator i = node.subvolume_begin(); i != node.subvolume_end(); ++i)
    {
      //root_bbox.merge( i->volume.bbox() );  // use aabb -> internal obb's might overlap!
      gpucast::math::oriented_boundingbox<ocnode::point_type>     obb = i->volume->obbox_by_mesh( gpucast::math::partial_derivative_policy<ocnode::point_type>(true));
      gpucast::math::axis_aligned_boundingbox<ocnode::point_type> aabb = obb.aabb();
      root_bbox.merge ( aabb );
    }

    // add a 2% offset around total bounding box
    root_bbox.scale(0.02f);

    // set bounding box
    node.boundingbox ( root_bbox );
  }

  // subdivide node
  //if ( !node.empty() && node.depth() < _max_depth )
  if ( ocsplit_by_volumecount(_max_depth, 20)(node) )
  {
    // octreemize bounding box
    std::list<node::boundingbox_ptr> ocsplit;
    node.boundingbox()->uniform_split(ocsplit);

    std::size_t ocsplit_id = 0;

    // create children nodes and apply their size
    for(node::boundingbox_ptr const& abox : ocsplit)
    {
      std::shared_ptr<ocnode::boundingbox_type> b = boost::dynamic_pointer_cast<ocnode::boundingbox_type>(abox);

      // apply axis aligned bounding box to new child node and add as child
      ocnode::pointer child_ptr(new ocnode(node.shared_from_this(), node.depth() + 1, *b, ocsplit_id++ ));
      node.add(child_ptr);

      for (node::const_subvolume_iterator i = node.subvolume_begin(); i != node.subvolume_end(); ++i)
      {
        // if obb of subvolume overlaps childnode's bbox add subvolume to child
        //if ( i->volume.obbox_by_mesh(gpucast::math::greedy_policy<node::point_type>()).overlap(*b) )
        gpucast::math::oriented_boundingbox<ocnode::point_type> ocnode_obb = *b;
        gpucast::math::oriented_boundingbox<ocnode::point_type> volume_obb = i->volume->obbox_by_mesh(gpucast::math::partial_derivative_policy<node::point_type>(true));

        //if ( volume_obb.overlap(ocnode_obb) )
        if ( ocnode_obb.overlap(volume_obb)  )
        //if ( i->volume.obbox_by_mesh(gpucast::math::partial_derivative_policy<node::point_type>(true)).overlap(*b) )
        //if ( i->volume.obbox_by_mesh<gpucast::math::random_policy>().overlap(b) )
        //if ( i->volume.bbox().overlap(b) )
        {
          child_ptr->add(*i);
        }
      }
    }
    //node.remove_all_data();
  }

  std::for_each(node.begin(), node.end(), std::bind(&node::accept, std::placeholders::_1, *this));
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ octree_refinement::node_ptr
octree_refinement::create_root () const
{
  return node_ptr( new ocnode() );
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
octree_refinement::visit ( ocnode& )
{
  std::cout << "try to refine ocnode bei octreemizing..." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
obbtree_refinement::visit ( ocnode& n )
{
  if ( n.has_children() ) // has children -> traverse to leaves
  {
    std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, std::ref(*this)));
  } else { // reached leaf that will be split

  }
}


///////////////////////////////////////////////////////////////////////////////
draw_traversal::draw_traversal(gpucast::gl::matrix4x4<float> const& mvp)
  : _mvp(mvp)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
draw_traversal::visit ( ocnode& n )
{
  n.draw(_mvp);
  std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, *this));
}


///////////////////////////////////////////////////////////////////////////////
info_traversal::info_traversal()
: _nodes        (0),
  _max_depth    (0),
  _volume       (0.0),
  _surface      (0.0)
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
info_traversal::visit ( ocnode& n )
{
  ++_nodes;
  _max_depth = std::max(_max_depth, n.depth());

  _volume  += n.volume();
  _surface += n.surface();

  std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, std::ref(*this)));
  std::for_each(n.subvolume_begin(),
                n.subvolume_end(),
                [&] ( beziersubvolume const& b )
                {
                  for ( beziervolume::datamap::const_iterator i = b.volume->data_begin(); i != b.volume->data_end(); ++i )
                  {
                    beziervolume::boundingbox_type datalimits = i->second.bbox();

                    // first entry
                    if ( _minmax.find(i->first) == _minmax.end() )
                    {
                      _minmax[i->first] = datalimits; // use this bbox
                    } else {
                      _minmax[i->first].merge(datalimits); // merge with existing bbox
                    }
                  }
                } );

  // get total size
  if (!n.has_parent())
  {
    _minmax["size"] = *boost::dynamic_pointer_cast<ocnode::boundingbox_type >(n.boundingbox());
  }
}


///////////////////////////////////////////////////////////////////////////////
void
info_traversal::print ( std::ostream& os ) const
{
  os << "nodes : " << _nodes << std::endl;
  os << "depth : " << _max_depth << std::endl;
  os << "total surface : " << _surface << std::endl;
  os << "total volume : " << _volume << std::endl;

  for ( minmax_map::const_iterator i = _minmax.begin(); i != _minmax.end(); ++i )
  {
    std::string attribute_name = i->first;
    os << attribute_name << " : " << _minmax.at(attribute_name).min << " - " << _minmax.at(attribute_name).max << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////
node::point_type
info_traversal::min ( std::string const& attribute_name ) const
{
  return _minmax.at(attribute_name).min;
}


///////////////////////////////////////////////////////////////////////////////
node::point_type
info_traversal::max ( std::string const& attribute_name ) const
{
  return _minmax.at(attribute_name).max;
}


///////////////////////////////////////////////////////////////////////////////
info_traversal::minmax_map
info_traversal::get_minmax_map() const
{
  return _minmax;
}



///////////////////////////////////////////////////////////////////////////////
compute_depth_traversal::compute_depth_traversal(std::size_t start_depth)
: _current_depth(start_depth)
{}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
compute_depth_traversal::visit ( ocnode& o )
{
  o.depth(_current_depth);
  std::for_each(o.begin(), o.end(), std::bind(&node::accept, std::placeholders::_1, compute_depth_traversal(_current_depth + 1)));
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void tight_obb_traversal::visit ( ocnode& n )
{
  if ( n.has_children() ) {
    std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, std::ref(*this)));
  } else {
    for (node::subvolume_iterator s = n.subvolume_begin(); s != n.subvolume_end(); ++s)
    {
      std::list<node::point_type> tmp = s->volume->convexhull();
      std::copy(tmp.begin(), tmp.end(), std::back_inserter(_points));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
serialize_tree_dfs_traversal::visit ( ocnode& n )
{
  // gather general node information
  int nodetype  = n.has_children() ? 0 : 1;
  int id_limits = int( limitbuffer.empty() ? 0 : limitbuffer.begin()->second.size() );
  int ndata     = n.has_children() ? 0 : int( n.size() );
  int id_data   = int( volumelistbuffer.size() );

  // write redirection from parent node to this node
  indirection_map_type::const_iterator k = indirection_map.find(n.shared_from_this());
  if ( k != indirection_map.end() ) // found unresolved parent link
  {
    int node_index = int(treebuffer.size());
    treebuffer[int(k->second.first)][int(k->second.second)] = node_index; // write link from parent to child
  } else {
    // no parent found -> has to be root node -> do nothing
  }

  // write independent information about this node
  treebuffer.push_back ( gpucast::gl::vec4i( nodetype, id_limits, ndata, id_data ) );
  for ( std::string const& name : n.data() )
  {
    limitbuffer[name].push_back( determine_minimum(n, name) );
    limitbuffer[name].push_back( determine_maximum(n, name) );
  }

  // write dependent information about this node
  if ( n.has_children() )  // write information about children
  {
    if ( n.children() == 8 ) // requirement that ocsplit results in 8 children
    {
      for ( node::const_iterator i = n.begin(); i != n.end(); ++i)
      {
        std::shared_ptr<ocnode> node = boost::dynamic_pointer_cast<ocnode>(*i);

        //if (!node) throw std::runtime_error("Unresolved conflict! non-ocnode child!");
        if (!node) std::cerr << "Unresolved conflict! non-ocnode child!" << std::endl;

        std::size_t base_index = treebuffer.size() + node->id() / gpucast::gl::vec4<int>::size;
        std::size_t sub_index  = node->id() % gpucast::gl::vec4<int>::size;
        indirection_map.insert(std::make_pair(*i, std::make_pair(base_index, sub_index)));
      }
      treebuffer.push_back(gpucast::gl::vec4i()); // indirection indices are filled by children
      treebuffer.push_back(gpucast::gl::vec4i()); // indirection indices are filled by children
    } else {
      // handle different kind/amount of nodes
      //throw std::runtime_error("OBBNodes not handled yet.");
      std::cout << "OBBNodes not handled yet." << std::endl;
    }
  } else {

    for ( node::const_subvolume_iterator i = n.subvolume_begin(); i != n.subvolume_end(); ++i )
    {
      add_subvolume(*i);
    }
  }

  // visit all children
  std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, std::ref(*this)));
}


///////////////////////////////////////////////////////////////////////////////
void 
serialize_tree_dfs_traversal::add_subvolume ( beziersubvolume const& i )
{
  // volume already contained in buffer
  if ( volume_map.count(i.volume) != 0 ) {
    // use stored indices 
    volumelistbuffer.push_back( volume_map.find(i.volume)->second.first  );
    volumelistbuffer.push_back( volume_map.find(i.volume)->second.second );  
    return;
  }

  for ( beziervolume::datamap::const_iterator d = i.volume->data_begin(); d != i.volume->data_end(); ++d )
  {
    // allocate buffer
    databuffer[d->first];
    limitbuffer[d->first];
  }

  if ( databuffer.empty() )
  {
    //throw std::runtime_error("No data volume elements");
    std::cerr << "No data volume elements" << std::endl;
  }

  // compute necessary indices
  int volumelist_id    = int(volumelistbuffer.size());
  int volume_id        = int(volumebuffer.size());
  int surfacebuffer_id = int(surfacebuffer.size());
  int bboxbuffer_id    = int(boundingboxbuffer.size());
  int limitbuffer_id   = int(limitbuffer.begin()->second.size());

  // set indices in volumelistbuffer
  gpucast::gl::vec4i volumeinfo1 = gpucast::gl::vec4i ( volume_id, int(i.parent->degree_u() + 1), int(i.parent->degree_v() + 1), int(i.parent->degree_w() + 1) );
  gpucast::gl::vec4i volumeinfo2 = gpucast::gl::vec4i ( surfacebuffer_id, bboxbuffer_id, limitbuffer_id, 0 );
  volumelistbuffer.push_back( volumeinfo1 );
  volumelistbuffer.push_back( volumeinfo2 );

  // store main indices for reuse in other nodes
  volume_map.insert(std::make_pair(i.volume, std::make_pair(volumeinfo1, volumeinfo2)));

  // copy volume control points into volumebuffer
  for(beziervolume::point_type const& p : *(i.volume) )
  {
    volumebuffer.push_back  ( gpucast::gl::vec4f ( p.as_homogenous() ) );
  }

  // copy outer surfaces in order : umin, umax, vmin, vmax, wmin, wmax
  add_outer_surfaces(*(i.volume));

  // gather information related to boundingbox
  gpucast::math::partial_derivative_policy<beziervolume::point_type> obb_generator(true);
  gpucast::math::oriented_boundingbox<beziervolume::point_type> obb = i.volume->obbox_by_mesh(obb_generator);

  gpucast::math::oriented_boundingbox<beziervolume::point_type>::matrix_type orientation          = obb.orientation();
  gpucast::math::oriented_boundingbox<beziervolume::point_type>::matrix_type orientation_inverse  = orientation.inverse();

  // transpose for gpu
  orientation               = orientation.transpose();
  orientation_inverse       = orientation_inverse.transpose();

  // push orientation
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(orientation[0][0]), float(orientation[0][1]), float(orientation[0][2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(orientation[1][0]), float(orientation[1][1]), float(orientation[1][2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(orientation[2][0]), float(orientation[2][1]), float(orientation[2][2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( 0.0f, 0.0f, 0.0f, 1.0f ));

  // push inverse orientation
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(orientation_inverse[0][0]), float(orientation_inverse[0][1]), float(orientation_inverse[0][2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(orientation_inverse[1][0]), float(orientation_inverse[1][1]), float(orientation_inverse[1][2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(orientation_inverse[2][0]), float(orientation_inverse[2][1]), float(orientation_inverse[2][2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( 0.0f, 0.0f, 0.0f, 1.0f ));

  // push dimension and center
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(obb.low()[0]),    float(obb.low()[1]),    float(obb.low()[2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(obb.high()[0]),   float(obb.high()[1]),   float(obb.high()[2]), 0.0f ));
  boundingboxbuffer.push_back( gpucast::gl::vec4f( float(obb.center()[0]), float(obb.center()[1]), float(obb.center()[2]), 0.0f ));

  // compute parameter values of oob's corner's
  std::list<beziervolume::point_type> corners;
  std::vector<gpucast::gl::vec4f>            corner_uvw_values;
  obb.generate_corners( std::back_inserter (corners) );

  unsigned point_index = 0;
  std::for_each ( corners.begin(), corners.end(), [&point_index, &i, &corner_uvw_values] (beziervolume::point_type const& p) 
                                                      { 
                                                        // iterate from domain corner uvw in [0,1] to theoretical match in [-inf, inf]
                                                        beziervolume::point_type uvw_start  ( point_index%2, (point_index%4)/2, point_index/4 );
                                                        beziervolume::point_type uvw;
                                                        gpucast::math::newton_raphson() ( *i.volume, p, uvw_start, uvw );

                                                        // classify result
                                                        float const error_threshold = 10.0f;
                                                        if ( uvw.abs() < error_threshold )
                                                        {
                                                          // expect iteration to be converged
                                                          corner_uvw_values.push_back( gpucast::gl::vec4f ( float(uvw[0]), float(uvw[1]), float(uvw[2]), 0.0f ));
                                                        } else {
                                                          // iteration converged to point quite far away -> iteration failed -> use starting value
                                                          corner_uvw_values.push_back( gpucast::gl::vec4f ( float(uvw_start[0]), float(uvw_start[1]), float(uvw_start[2]), 0.0f ));
                                                        }

                                                        ++point_index;
                                                      } 
                );

  // push parameter corner mesh into obb-buffer
  std::copy(corner_uvw_values.begin(), corner_uvw_values.end(), std::back_inserter(boundingboxbuffer));

  // information: displacement etc.
  for ( beziervolume::datamap::const_iterator d = i.volume->data_begin(); d != i.volume->data_end(); ++d )
  {
    // name
    std::string const& name = d->first;

    // compute bounds
    beziervolume::boundingbox_type aabb = d->second.bbox();
    limitbuffer[name].push_back( gpucast::gl::vec4f( aabb.min ));
    limitbuffer[name].push_back( gpucast::gl::vec4f( aabb.max ));

    // points
    for (beziervolume::point_type const& p : d->second)
    {
      databuffer[name].push_back( gpucast::gl::vec4f ( p.as_homogenous() ) );
    }
  }
}




///////////////////////////////////////////////////////////////////////////////
void
serialize_tree_dfs_traversal::add_outer_surfaces ( beziervolume const& v )
{
  gpucast::math::beziersurface<beziervolume::point_type> umin = v.slice(0, 0);
  gpucast::math::beziersurface<beziervolume::point_type> umax = v.slice(0, v.degree_u() );
  gpucast::math::beziersurface<beziervolume::point_type> vmin = v.slice(1, 0);
  gpucast::math::beziersurface<beziervolume::point_type> vmax = v.slice(1, v.degree_v() );
  gpucast::math::beziersurface<beziervolume::point_type> wmin = v.slice(2, 0);
  gpucast::math::beziersurface<beziervolume::point_type> wmax = v.slice(2, v.degree_w() );

  for (beziervolume::point_type const& p : umin) {
    surfacebuffer.push_back( gpucast::gl::vec4f( p.as_homogenous() ) );
  }

  for (beziervolume::point_type const& p : umax) {
    surfacebuffer.push_back( gpucast::gl::vec4f( p.as_homogenous() ) );
  }

  for (beziervolume::point_type const& p : vmin) {
    surfacebuffer.push_back( gpucast::gl::vec4f( p.as_homogenous() ) );
  }

  for (beziervolume::point_type const& p : vmax) {
    surfacebuffer.push_back( gpucast::gl::vec4f( p.as_homogenous() ) );
  }

  for (beziervolume::point_type const& p : wmin) {
    surfacebuffer.push_back( gpucast::gl::vec4f( p.as_homogenous() ) );
  }

  for (beziervolume::point_type const& p : wmax) {
    surfacebuffer.push_back( gpucast::gl::vec4f( p.as_homogenous() ) );
  }
}


///////////////////////////////////////////////////////////////////////////////
beziervolume::point_type
serialize_tree_dfs_traversal::determine_minimum ( node& n, std::string const& name )
{
  bool init = false;
  beziervolume::point_type dmin;

  for ( node::const_subvolume_iterator i = n.subvolume_begin(); i != n.subvolume_end(); ++i ) {
    for ( beziervolume::datamap::const_iterator d = i->volume->data_begin(); d != i->volume->data_end(); ++d )
    {
      gpucast::math::axis_aligned_boundingbox<beziervolume::point_type> aabb = d->second.bbox();
      if ( !init &&
           name == d->first)
      {
        dmin = aabb.min;
        init = true;
      } else {
        dmin = elementwise_min(dmin, aabb.min);
      }
    }
  }

  return init ? dmin : beziervolume::point_type();
}


///////////////////////////////////////////////////////////////////////////////
beziervolume::point_type
serialize_tree_dfs_traversal::determine_maximum ( node& n, std::string const& name )
{
  bool init = false;
  beziervolume::point_type dmax;

  for ( node::const_subvolume_iterator i = n.subvolume_begin(); i != n.subvolume_end(); ++i ) {
    for ( beziervolume::datamap::const_iterator d = i->volume->data_begin(); d != i->volume->data_end(); ++d )
    {
      gpucast::math::axis_aligned_boundingbox<beziervolume::point_type> aabb = d->second.bbox();

      if ( !init &&
           name == d->first)
      {
        dmax = aabb.max;
        init = true;
      } else {
        dmax = elementwise_max(dmax, aabb.max);
      }
    }
  }

  return init ? dmax : beziervolume::point_type();
}

#endif

} // namespace gpucast
