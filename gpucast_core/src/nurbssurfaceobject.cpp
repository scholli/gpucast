/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbssurfaceobject.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/nurbssurfaceobject.hpp"

// header, system
#include <iostream>



namespace gpucast {

  ////////////////////////////////////////////////////////////////////////////////
  nurbssurfaceobject::nurbssurfaceobject()
    : _surfaces()
  {}


  ////////////////////////////////////////////////////////////////////////////////
  nurbssurfaceobject::~nurbssurfaceobject()
  {}


////////////////////////////////////////////////////////////////////////////////
  void
  nurbssurfaceobject::add ( nurbssurface const& nrbs )
  {
    _surfaces.push_back(nrbs);
  }


  ////////////////////////////////////////////////////////////////////////////////
  void nurbssurfaceobject::print(std::ostream& os) const
  {
    os << "Non-Rational NURBS in Object : " << std::endl;
    for (std::vector<nurbssurface>::const_iterator i = _surfaces.begin(); i < _surfaces.end(); ++i) {
      os << (*i);
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  nurbssurfaceobject::const_iterator
  nurbssurfaceobject::begin () const
  {
    return _surfaces.begin();
  }

  ////////////////////////////////////////////////////////////////////////////////
  nurbssurfaceobject::const_iterator
  nurbssurfaceobject::end () const
  {
    return _surfaces.end();
  }

  ////////////////////////////////////////////////////////////////////////////////
  nurbssurfaceobject::iterator
  nurbssurfaceobject::begin()
  {
    return _surfaces.begin();
  }

  ////////////////////////////////////////////////////////////////////////////////
  nurbssurfaceobject::iterator
  nurbssurfaceobject::end()
  {
    return _surfaces.end();
  }

  ////////////////////////////////////////////////////////////////////////////////
  std::size_t
  nurbssurfaceobject::surfaces() const
  {
    return _surfaces.size();
  }


  ////////////////////////////////////////////////////////////////////////////////
  std::size_t
  nurbssurfaceobject::trimcurves() const
  {
    std::size_t ntrimcurves = 0;

    for (const_iterator i = _surfaces.begin(); i != _surfaces.end(); ++i)
    {
      for ( auto loop = i->trimloops().begin(); loop != i->trimloops().end(); ++loop )
      {
        ntrimcurves += std::size_t(loop->size());
      }
    }

    return ntrimcurves;
  }


} // namespace gpucast
