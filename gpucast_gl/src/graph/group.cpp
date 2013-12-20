/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : group.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/graph/group.hpp"

// header, system
#include <boost/foreach.hpp>
#include <boost/mem_fn.hpp>
#include <boost/bind.hpp>

// header, project
#include <gpucast/gl/graph/visitor.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
group::group()
: node      (),
  _matrix   (),
  _children ()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ group::~group()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
group::visit ( visitor const& v )
{
  v.accept(*this);
  std::for_each(begin(), end(), std::bind(&node::visit, std::placeholders::_1, std::ref(v)));
}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void 
group::compute_bbox ()
{
  BOOST_FOREACH(node_ptr_t n, _children ) 
  {
    n->compute_bbox();
    _bbox.merge(n->bbox());
  }
}


///////////////////////////////////////////////////////////////////////////////
void        
group::add ( group::node_ptr_t child )
{
  _children.insert(child);
}


///////////////////////////////////////////////////////////////////////////////
void        
group::remove ( group::node_ptr_t child )
{
  _children.erase(child);
}


///////////////////////////////////////////////////////////////////////////////
std::size_t             
group::children () const
{
  return _children.size();
}


///////////////////////////////////////////////////////////////////////////////
group::iterator                
group::begin ()
{
  return _children.begin();
}


///////////////////////////////////////////////////////////////////////////////
group::iterator                
group::end ()
{
  return _children.end();
}


///////////////////////////////////////////////////////////////////////////////
group::const_iterator          
group::begin () const
{
  return _children.begin();
}


///////////////////////////////////////////////////////////////////////////////
group::const_iterator          
group::end () const
{
  return _children.end();
}


///////////////////////////////////////////////////////////////////////////////
void 
group::set_transform ( matrix4f const& matrix )
{
  _matrix = matrix;
}


///////////////////////////////////////////////////////////////////////////////
void 
group::translate ( vec3f const& t)
{
  _matrix *= make_translation(t[0], t[1], t[2]);
}


///////////////////////////////////////////////////////////////////////////////
void 
group::rotate ( float alpha, vec3f const& axis)
{
  _matrix *= make_rotation_x(alpha * axis[0]) *
             make_rotation_y(alpha * axis[1]) *
             make_rotation_z(alpha * axis[2]);
}

///////////////////////////////////////////////////////////////////////////////
void 
group::scale ( vec3f const& scale )
{
  _matrix *= make_scale(scale[0], scale[1], scale[2]);
}

} } // namespace gpucast / namespace gl
