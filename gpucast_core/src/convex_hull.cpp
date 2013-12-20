/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : convex_hull.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// header i/f
#include "gpucast/core/convex_hull.hpp"
#include <gpucast/core/convex_hull_impl.hpp>

// header, system
#include <algorithm>
#include <iterator>

// header, project

namespace gpucast {


////////////////////////////////////////////////////////////////////////////////
convex_hull::convex_hull()
  : _vertices   (),
    _parameter  (),
    _indices    ()
{}


////////////////////////////////////////////////////////////////////////////////
convex_hull::convex_hull(convex_hull const& cpy)
  : _vertices   ( cpy._vertices ),
    _parameter  ( cpy._parameter),
    _indices    ( cpy._indices  )
{}



////////////////////////////////////////////////////////////////////////////////
convex_hull::~convex_hull()
{}



////////////////////////////////////////////////////////////////////////////////
convex_hull&
convex_hull::operator=(convex_hull const& rhs)
{
  convex_hull tmp(rhs);
  swap(tmp);
  return *this;
}



////////////////////////////////////////////////////////////////////////////////
void convex_hull::swap(convex_hull& swp)
{
  std::swap(_vertices , swp._vertices );
  std::swap(_parameter, swp._parameter);
  std::swap(_indices  , swp._indices  );
}

//////////////////////////////////////////////////////////////////////////////
void
convex_hull::set(vec3d_iterator       p_beg,
                 vec3d_iterator       p_end,
                 const_vec4f_iterator c_beg,
                 const_vec4f_iterator c_end)
{
  // make sure arrays have same length
  assert(std::distance(p_beg, p_end) == std::distance(c_beg, c_end));

  // clear old chull
  _vertices.clear();
  _indices.clear();

  convex_hull_compute<3, gpucast::gl::vec3d>(&(*p_beg)[0],
                                      std::distance(p_beg, p_end),
                                      std::back_inserter(_vertices),
                                      std::back_inserter(_indices),
                                      0);

  std::copy(c_beg, c_end, std::back_inserter(_parameter));
}


//////////////////////////////////////////////////////////////////////////////
std::size_t
convex_hull::size() const
{
  return _vertices.size();
}

//////////////////////////////////////////////////////////////////////////////
void
convex_hull::clear ()
{
  _vertices.clear();
  _indices.clear();
  _parameter.clear();
}


////////////////////////////////////////////////////////////////////////////////
void
convex_hull::print(std::ostream& os) const
{
  std::for_each(_indices.begin(), _indices.end(), triangle_printer<gpucast::gl::vec3d>(_vertices, os));
}


//////////////////////////////////////////////////////////////////////////////
void
convex_hull::merge(convex_hull const& cpy)
{
  unsigned offset = unsigned(_vertices.size());

  std::copy(cpy._vertices.begin(),  cpy._vertices.end(),   std::back_inserter(_vertices));
  std::copy(cpy._parameter.begin(), cpy._parameter.end(),  std::back_inserter(_parameter));

  std::transform(cpy._indices.begin(), cpy._indices.end(),
		             std::back_inserter(_indices),
                 [&] ( int i ) { return i + offset; } );
}


////////////////////////////////////////////////////////////////////////////////
convex_hull::const_vec3d_iterator 
convex_hull::vertices_begin () const
{
  return _vertices.begin();
}


////////////////////////////////////////////////////////////////////////////////
convex_hull::const_vec3d_iterator 
convex_hull::vertices_end () const
{
  return _vertices.end();
}


////////////////////////////////////////////////////////////////////////////////
convex_hull::const_vec4f_iterator 
convex_hull::uv_begin () const
{
  return _parameter.begin();
}


////////////////////////////////////////////////////////////////////////////////
convex_hull::const_vec4f_iterator 
convex_hull::uv_end () const
{
  return _parameter.end();
}


////////////////////////////////////////////////////////////////////////////////
convex_hull::const_index_iterator 
convex_hull::indices_begin () const
{
  return _indices.begin();
}


////////////////////////////////////////////////////////////////////////////////
convex_hull::const_index_iterator 
convex_hull::indices_end () const
{
  return _indices.end();
}


////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, convex_hull const& rhs)
{
  rhs.print(os);
  return os;
}

} // namespace gpucast
