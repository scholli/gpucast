/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziervolumeobject.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/beziervolumeobject.hpp"

// header, project
#include <gpucast/volume/nurbsvolumeobject.hpp>

#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/primitives/cube.hpp>

#include <functional>

#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::beziervolumeobject ( beziervolumeobject::volume_renderer_ptr const& renderer, unsigned uid )
  : _nurbsobject ( ),
    _uid         ( uid ),
    _volumes     ( )
{} 


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::~beziervolumeobject ()
{}

////////////////////////////////////////////////////////////////////////////////
void                              
beziervolumeobject::swap ( beziervolumeobject& other )
{
  _nurbsobject.swap(other._nurbsobject);

  std::swap(_uid, other._uid);

  _volumes.swap(other._volumes);
}

////////////////////////////////////////////////////////////////////////////////
void 
beziervolumeobject::add ( element_type const& v )
{
  _volumes.push_back( element_type(v) );

  element_type& p = _volumes.back();

  while (p.degree_u() <= 1) p.elevate_u();
  while (p.degree_v() <= 1) p.elevate_v();
  while (p.degree_w() <= 1) p.elevate_w();

  for (element_type::attribute_volume_map::iterator d = p.data_begin(); d != p.data_end(); ++d)
  {
    while (d->second.degree_u() <= 1) d->second.elevate_u();
    while (d->second.degree_v() <= 1) d->second.elevate_v();
    while (d->second.degree_w() <= 1) d->second.elevate_w();
  }
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolumeobject::parent ( beziervolumeobject::nurbsvolumeobject_ptr const& n )
{
  _nurbsobject = n;
}


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::nurbsvolumeobject_ptr const&                              
beziervolumeobject::parent ( ) const
{
  return _nurbsobject;
}


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::boundingbox_type      
beziervolumeobject::bbox () const
{
  element_type::boundingbox_type bbox;

  if ( _volumes.empty() ) {
    return bbox;
  } else {
    bbox = _volumes.front().bbox();
    for ( element_type const& v : _volumes)
    {
      bbox.merge(v.bbox());
    }
  }

  return bbox;
} 


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::const_iterator 
beziervolumeobject::begin () const
{
  return _volumes.begin();
}


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::const_iterator 
beziervolumeobject::end () const
{
  return _volumes.end();
}


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::iterator 
beziervolumeobject::begin ()
{
  return _volumes.begin();
}


////////////////////////////////////////////////////////////////////////////////
beziervolumeobject::iterator 
beziervolumeobject::end ()
{
  return _volumes.end();
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
beziervolumeobject::size () const
{
  return _volumes.size();
}


////////////////////////////////////////////////////////////////////////////////
unsigned                          
beziervolumeobject::uid () const
{
  return _uid;
}


////////////////////////////////////////////////////////////////////////////////
void                  
beziervolumeobject::clear ()
{
  _nurbsobject.reset();

  // clear data
  _volumes.clear();

  _uid = 0;
}


////////////////////////////////////////////////////////////////////////////////
void 
beziervolumeobject::write ( std::ostream& os ) const
{
  // 1. parent volume
  _nurbsobject->write(os);

  // 2. UID
  os.write( reinterpret_cast<char const*> (&_uid),               sizeof(unsigned int) );

  // 3. bezier volume elements
  std::size_t elements = _volumes.size();
  os.write( reinterpret_cast<char const*> (&elements), sizeof(std::size_t) );
  std::for_each(_volumes.begin(), _volumes.end(), std::bind(&beziervolume::write, std::placeholders::_1, std::ref(os)));
};


////////////////////////////////////////////////////////////////////////////////
void 
beziervolumeobject::read ( std::istream& is )
{
  _volumes.clear();

  // 1. parent nurbs volume
  _nurbsobject.reset(new nurbsvolumeobject);
  _nurbsobject->read(is);

  // 2. UID
  is.read( reinterpret_cast<char*> (&_uid),               sizeof ( unsigned int ) );

  std::size_t elements;
  is.read( reinterpret_cast<char*> (&elements), sizeof ( std::size_t ) );
  
  _volumes.resize(elements);
  std::for_each(_volumes.begin(), _volumes.end(), std::bind(&beziervolume::read, std::placeholders::_1, std::ref(is)));
}


} // namespace gpucast
