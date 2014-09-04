/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : ocnode.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/ocnode.hpp"

#include <gpucast/volume/obb.hpp>
#include <gpucast/volume/isosurface/octree/nodevisitor.hpp>

#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
ocnode::ocnode ()
: node                 (),
  _bbox                (),
  _contains_outer_face ( false ),
  _range               (),
  _faces               (),
  _id                  ( 0 )
{}


////////////////////////////////////////////////////////////////////////////////
ocnode::ocnode ( pointer const&           parent,
                 std::size_t              depth,
                 std::size_t              id )
: node                 ( parent, depth, id ),
  _bbox                (),
  _contains_outer_face ( false ),
  _range               (),
  _faces               (),
  _id                  ( id )
{}


////////////////////////////////////////////////////////////////////////////////
ocnode::ocnode ( pointer const&           parent,
                 std::size_t              depth,
                 boundingbox_type const&  bbox,
                 std::size_t              id )
: node                 ( parent, depth, id ),
  _bbox                ( bbox ),
  _contains_outer_face ( false ),
  _range               (),
  _faces               (),
  _id                  ( id )
{}


////////////////////////////////////////////////////////////////////////////////
ocnode::~ocnode()
{}


////////////////////////////////////////////////////////////////////////////////
node::boundingbox_type const&    
ocnode::boundingbox ( ) const
{
  return _bbox;
}


////////////////////////////////////////////////////////////////////////////////
void                  
ocnode::boundingbox ( boundingbox_type const& b )
{
  _bbox = b;
}

////////////////////////////////////////////////////////////////////////////////
ocnode::face_iterator
ocnode::face_begin ()
{
  return _faces.begin();
}


////////////////////////////////////////////////////////////////////////////////
ocnode::face_iterator
ocnode::face_end ()
{
  return _faces.end();
}


////////////////////////////////////////////////////////////////////////////////
ocnode::const_face_iterator
ocnode::face_begin () const
{
  return _faces.begin();
}


////////////////////////////////////////////////////////////////////////////////
ocnode::const_face_iterator
ocnode::face_end () const
{
  return _faces.end();
}


////////////////////////////////////////////////////////////////////////////////
void
ocnode::add_face ( face_ptr const& f )
{
  _faces.push_back(f);
}


////////////////////////////////////////////////////////////////////////////////
void
ocnode::clear_faces ()
{
  _faces.clear();
}


////////////////////////////////////////////////////////////////////////////////
ocnode::interval_t const&
ocnode::range () const
{
  return _range;
}


////////////////////////////////////////////////////////////////////////////////
void
ocnode::range ( interval_t const& r )
{
  _range = r;
}


////////////////////////////////////////////////////////////////////////////////
bool                                  
ocnode::contains_outer_face () const
{
  return _contains_outer_face;
}


////////////////////////////////////////////////////////////////////////////////
void                                  
ocnode::contains_outer_face ( bool b )
{
  _contains_outer_face = b;
}


////////////////////////////////////////////////////////////////////////////////
bool                                  
ocnode::empty () const
{
  return _faces.empty();
}


////////////////////////////////////////////////////////////////////////////////
std::size_t                           
ocnode::faces () const
{
  return _faces.size();
}


////////////////////////////////////////////////////////////////////////////////
void                    
ocnode::compute_bbox_from_children ()
{
  std::vector<node::point_type> corners;
  for (node::nodecontainer::const_iterator p = begin(); p != end(); ++p)
  {
    (*p)->boundingbox().generate_corners(std::back_inserter(corners)); 
  }

  boundingbox( boundingbox_type (corners.begin(), corners.end()));
}


////////////////////////////////////////////////////////////////////////////////
void                    
ocnode::compute_bbox_from_data ()
{
  throw std::runtime_error("not implemented");
}



////////////////////////////////////////////////////////////////////////////////
/* virtual */ node::value_type 
ocnode::volume () const
{
  return _bbox.volume();
}


////////////////////////////////////////////////////////////////////////////////
/* virtual */ node::value_type  
ocnode::surface () const
{
  return _bbox.surface();
}


////////////////////////////////////////////////////////////////////////////////
void                  
ocnode::accept ( nodevisitor const& visitor )
{
  visitor.visit(*this);
}

////////////////////////////////////////////////////////////////////////////////
/* virtual */ void ocnode::draw(gpucast::math::matrix4x4<float> const& mvp)
{

}

////////////////////////////////////////////////////////////////////////////////
void                  
ocnode::print ( std::ostream& os) const
{
  node::print(os);
  _bbox.print(os);
  os << " range: " << _range.minimum() << " - " << _range.maximum();
  os << " faces: " << _faces.size();  
  os << " id: " << _id;
}


////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, ocnode const& node)
{
  node.print(os);
  return os;
}

} // namespace gpucast
