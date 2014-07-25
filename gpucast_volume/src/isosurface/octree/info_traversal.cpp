/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : info_traversal.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/info_traversal.hpp"

// header, system

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>


namespace gpucast {

  struct info_traversal::_impl_t 
  {
    _impl_t()
      : inner_nodes (0),
        outer_nodes (0),
        depth       (0),
        faces       (0),
        max_faces   (0),
        empty_nodes (0),
        minmax      (0.f, 0.f)
    {}

    std::size_t           inner_nodes;  
    std::size_t           outer_nodes;  
    std::size_t           depth;
    std::size_t           faces;
    std::size_t           max_faces;
    std::size_t           empty_nodes;
    gpucast::math::interval<float>  minmax;
  };

///////////////////////////////////////////////////////////////////////////////
info_traversal::info_traversal()
: _impl        ( new _impl_t )
{}


///////////////////////////////////////////////////////////////////////////////
info_traversal::~info_traversal()
{
  delete _impl;
}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
info_traversal::visit ( ocnode& n ) const
{
  if ( n.has_children() ) { 
    ++_impl->inner_nodes;
  } else {
    ++_impl->outer_nodes;
  }

  _impl->depth = std::max(_impl->depth, n.depth());

  // traverse children
  std::for_each(n.begin(), n.end(), std::bind(&node::accept, std::placeholders::_1, std::ref(*this)));

  if ( n.faces() == 0 ) {
    ++_impl->empty_nodes;
  } else {
    if ( _impl->max_faces < n.faces() )
    {
      _impl->max_faces = n.faces();
    }
  }

  // count leave information
  std::for_each(n.face_begin(),
                n.face_end(),
                [&] ( face_ptr const& f )
                {
                  ++_impl->faces;
                  if ( _impl->minmax.length() == 0.0f ) 
                  {
                    _impl->minmax = f->attribute_range;
                  } else {
                    _impl->minmax.merge ( f->attribute_range );
                  }
                } );
}


///////////////////////////////////////////////////////////////////////////////
void
info_traversal::print ( std::ostream& os ) const
{
  os << "inner nodes : " << _impl->inner_nodes << std::endl;
  os << "outer nodes : " << _impl->outer_nodes << std::endl;
  os << "empty nodes : " << _impl->empty_nodes << std::endl;
  os << "depth : " << _impl->depth << std::endl;
  os << "faces : " << _impl->faces << std::endl;
  os << "max faces per node" << _impl->max_faces << std::endl;
  os << "average faces per leave : " << float(_impl->faces) / _impl->outer_nodes << std::endl;
  os << "attribute range : " << _impl->minmax << std::endl;
}


} // namespace gpucast
