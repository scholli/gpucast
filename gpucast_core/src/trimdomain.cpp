/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header, i/f
#include "gpucast/core/trimdomain.hpp"

// header, system
#include <iterator>
#include <limits>

// header, project
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/algorithm/bezierclipping2d.hpp>

using namespace gpucast::math;

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  trimdomain::trimdomain()
    : _trimloops    ( ),
      _type         ( false ),
      _nurbsdomain  ( point_type(0,0), point_type(1,1) )
  {}


  /////////////////////////////////////////////////////////////////////////////
  trimdomain::~trimdomain()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void
  trimdomain::swap(trimdomain& swp)
  {
    std::swap(_trimloops, swp._trimloops);
    std::swap(_type, swp._type);
    std::swap(_nurbsdomain, swp._nurbsdomain);
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  trimdomain::add(contour_type const& bc)
  {
    _trimloops.push_back(bc);
  }


  /////////////////////////////////////////////////////////////////////////////
  std::size_t
  trimdomain::size() const
  {
    std::size_t curves = 0;
    for ( auto l = _trimloops.begin(); l != _trimloops.end(); ++l )
    {
      curves += l->size();
    }
    return curves;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool        
  trimdomain::empty() const
  {
    return _trimloops.empty();
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  trimdomain::nurbsdomain ( bbox_type const& n)
  {
    _nurbsdomain = n;
  }


  /////////////////////////////////////////////////////////////////////////////
  trimdomain::bbox_type const&
  trimdomain::nurbsdomain ( ) const
  {
    return _nurbsdomain;
  }


  /////////////////////////////////////////////////////////////////////////////
  bool
  trimdomain::type() const
  {
    return _type;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  trimdomain::type(bool inner)
  {
    _type = inner;
  }


  /////////////////////////////////////////////////////////////////////////////
  trimdomain::curve_container       
  trimdomain::curves () const
  {
    curve_container curves;
    for ( contour_type const& loop : _trimloops )
    {
      std::copy ( loop.begin(), loop.end(), std::back_inserter ( curves ) );
    }
    return curves;
  }
  

  /////////////////////////////////////////////////////////////////////////////
  std::size_t           
  trimdomain::loop_count () const
  {
    return _trimloops.size();
  }

  /////////////////////////////////////////////////////////////////////////////
  std::size_t               
  trimdomain::max_degree() const {
    std::size_t maxdegree = 0;

    curve_container curves;
    for (contour_type const& loop : _trimloops) {
      for (auto curve = loop.begin(); curve != loop.end(); ++curve) {
        maxdegree = std::max(maxdegree, (*curve)->degree());
      }
    }
    return maxdegree;
  }


  /////////////////////////////////////////////////////////////////////////////
  trimdomain::trimloop_container const& 
  trimdomain::loops () const
  {
    return _trimloops;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  trimdomain::print(std::ostream& os) const
  {
    std::vector<curve_ptr> all_curves = curves();

    os << "nurbs domain total range : " << _nurbsdomain << std::endl;
    os << "# trimming curves : " << all_curves.size() << std::endl;
    std::for_each(all_curves.begin(), all_curves.end(), std::bind(&gpucast::math::beziercurve2d::print, std::placeholders::_1, std::ref(os),""));
  }


  /////////////////////////////////////////////////////////////////////////////
  std::ostream&
  operator<<(std::ostream& os, trimdomain const& t)
  {
    t.print(os);
    return os;
  }

} // namespace gpucast
