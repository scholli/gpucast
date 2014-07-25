/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : node.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/node.hpp"

namespace gpucast {


////////////////////////////////////////////////////////////////////////////////
node::node ()
: _parent       (),
  _children     (),
  _depth        ( 0 ),
  _id           ( 0 )
{}


////////////////////////////////////////////////////////////////////////////////
node::node ( pointer const& parent, std::size_t depth, std::size_t id )
: _parent       ( parent ),
  _children     (),
  _depth        ( depth ),
  _id           ( id )
{}


////////////////////////////////////////////////////////////////////////////////
node::~node()
{}


////////////////////////////////////////////////////////////////////////////////
bool
node::has_children () const
{
  return !_children.empty();
}


////////////////////////////////////////////////////////////////////////////////
bool
node::has_parent () const
{
  return _parent != 0;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
node::children ( ) const
{
  return _children.size();
}


////////////////////////////////////////////////////////////////////////////////
std::vector<node_ptr> const&
node::get_children () const
{
  return _children;
}

////////////////////////////////////////////////////////////////////////////////
void
node::clear ()
{
  _children.clear();
}


////////////////////////////////////////////////////////////////////////////////
void
node::add_node ( node_ptr const& n )
{
  _children.push_back(n);
}


////////////////////////////////////////////////////////////////////////////////
void
node::depth ( std::size_t d )
{
  _depth = d;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
node::depth () const
{
  return _depth;
}


////////////////////////////////////////////////////////////////////////////////
node::iterator
node::begin ()
{
  return _children.begin();
}


////////////////////////////////////////////////////////////////////////////////
node::iterator
node::end ()
{
  return _children.end();
}


////////////////////////////////////////////////////////////////////////////////
node::const_iterator
node::begin () const
{
  return _children.begin();
}


////////////////////////////////////////////////////////////////////////////////
node::const_iterator
node::end () const
{
  return _children.end();
}


////////////////////////////////////////////////////////////////////////////////
node::pointer const&
node::parent ( ) const
{
  return _parent;
}


////////////////////////////////////////////////////////////////////////////////
void
node::parent ( pointer const& p )
{
  _parent = p;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
node::id ( ) const
{
  return _id;
}


////////////////////////////////////////////////////////////////////////////////
void
node::id ( std::size_t i )
{
  _id = i;
}



////////////////////////////////////////////////////////////////////////////////
void
node::print ( std::ostream& os) const
{
  os << "parent     : "       << _parent << std::endl;
  os << "# children : "       << _children.size() << std::endl;
  os << "depth      : "       << _depth << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, node const& node)
{
  node.print(os);
  return os;
}

} // namespace gpucast
