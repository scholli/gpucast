/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface/fragment/renderbin.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/fragment/renderbin.hpp"

// header, system

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
renderbin::renderbin ( )
  : _range    ( -std::numeric_limits<value_type>::max(), std::numeric_limits<value_type>::max() ),
    _chunks   (),
    _baseindex(0),
    _indices  (0)
{}


////////////////////////////////////////////////////////////////////////////////
renderbin::renderbin ( interval_type const& range )
  : _range    ( range ),
    _chunks   (),
    _baseindex(0),
    _indices  (0)
{}


////////////////////////////////////////////////////////////////////////////////
renderbin::renderbin ( renderbin const& other )
  : _range    ( other._range ),
    _chunks   ( other._chunks ),
    _baseindex( other._baseindex ),
    _indices  ( other._indices )
{}


////////////////////////////////////////////////////////////////////////////////
renderbin& 
renderbin::operator= ( renderbin const& rhs )
{
  renderbin tmp ( rhs );
  tmp.swap ( *this );
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
void                        
renderbin::swap ( renderbin& other )
{
  std::swap    ( _range, other._range );
  std::swap    ( _baseindex, other._baseindex );
  std::swap    ( _indices, other._indices );

  _chunks.swap ( other._chunks );
}


////////////////////////////////////////////////////////////////////////////////
renderbin::~renderbin()
{}


////////////////////////////////////////////////////////////////////////////////
renderbin::interval_type const&          
renderbin::range () const
{
  return _range;
}


////////////////////////////////////////////////////////////////////////////////
void                          
renderbin::range ( interval_type const& range )
{
  _range = range;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t 
renderbin::chunks() const
{
  return _chunks.size();
}


////////////////////////////////////////////////////////////////////////////////
renderbin::renderchunk_const_iterator 
renderbin::begin () const
{
  return _chunks.begin();
}


////////////////////////////////////////////////////////////////////////////////
renderbin::renderchunk_const_iterator 
renderbin::end () const
{
  return _chunks.end();
}


////////////////////////////////////////////////////////////////////////////////
int   
renderbin::indices () const
{
  return _indices;
}


////////////////////////////////////////////////////////////////////////////////
int   
renderbin::baseindex () const
{
  return _baseindex;
}


////////////////////////////////////////////////////////////////////////////////
void  
renderbin::indices ( int s )
{
  _indices = s;
}


////////////////////////////////////////////////////////////////////////////////
void  
renderbin::baseindex ( int i )
{
  _baseindex = i;
}


////////////////////////////////////////////////////////////////////////////////
void  
renderbin::insert ( renderchunk_ptr const& r )
{
  //assert ( _range.overlap(r->range) );

  _chunks.push_back(r);
}

////////////////////////////////////////////////////////////////////////////////
bool operator==( renderbin const& lhs, renderbin const& rhs )
{
  return lhs.range() == rhs.range();
}
////////////////////////////////////////////////////////////////////////////////
bool operator!=( renderbin const& lhs, renderbin const& rhs )
{
  return !(lhs == rhs);
}
////////////////////////////////////////////////////////////////////////////////
bool operator> ( renderbin const& lhs, renderbin const& rhs )
{
  return lhs.range().minimum() > rhs.range().minimum();
}

////////////////////////////////////////////////////////////////////////////////
bool operator< ( renderbin const& lhs, renderbin const& rhs )
{
  return lhs.range().minimum() < rhs.range().minimum();
}

} // namespace gpucast

