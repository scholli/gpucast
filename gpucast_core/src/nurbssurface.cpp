/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbssurface.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#include "gpucast/core/nurbssurface.hpp"
// header, system
#include <cassert> // std::assert

// header, project
#include <gpucast/math/parametric/nurbscurve.hpp>
#include <gpucast/math/parametric/nurbssurface.hpp>
#include <gpucast/math/parametric/point.hpp>

#include <gpucast/core/gpucast.hpp>

namespace gpucast {

  //////////////////////////////////////////////////////////////////////////////
  nurbssurface::nurbssurface()
    :  gpucast::math::nurbssurface< gpucast::math::point3d> ( ),
      _trimloops ( ),
      _trimtype   ( true )
  {}


  //////////////////////////////////////////////////////////////////////////////
  nurbssurface::nurbssurface(nurbssurface const& rhs)
    :  gpucast::math::nurbssurface< gpucast::math::point3d> ( rhs ),
      _trimloops  ( rhs._trimloops ),
      _trimtype   ( rhs._trimtype )
  {}

  //////////////////////////////////////////////////////////////////////////////
  nurbssurface::nurbssurface(math::nurbssurface3d const& ns)
    : gpucast::math::nurbssurface<gpucast::math::point3d>(ns),
      _trimloops(),
      _trimtype(true)
  {}

  //////////////////////////////////////////////////////////////////////////////
  nurbssurface::~nurbssurface()
  {}


  //////////////////////////////////////////////////////////////////////////////
  void
  nurbssurface::swap(nurbssurface& rhs)
  {
     gpucast::math::nurbssurface< gpucast::math::point3d>::swap(rhs);

    std::swap(_trimloops, rhs._trimloops);
    std::swap(_trimtype  , rhs._trimtype);

  }


  //////////////////////////////////////////////////////////////////////////////
  nurbssurface&
  nurbssurface::operator=(nurbssurface const& rhs)
  {
    nurbssurface tmp(rhs);
    swap(tmp);
    return *this;
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  nurbssurface::add(curve_container const& tc)
  {
    _trimloops.push_back(tc);
  }

  //////////////////////////////////////////////////////////////////////////////
  void                      
  nurbssurface::normalize()
  {
    auto urange = umax() - umin();
    auto vrange = vmax() - vmin();

    if (urange > vrange) { // rescale v to u
      auto factor = urange / vrange;

      // normalize knotvector
      knotvector_type scaled_knotvector_v;
      for (auto k : knotvector_v()) {
        scaled_knotvector_v.push_back(vmin() + (k - vmin()) * factor);
      }
      knotvector_v(scaled_knotvector_v.begin(), scaled_knotvector_v.end());

      // normalize trim curves 
      for (auto& tl : _trimloops) {
        for (auto& c : tl) {
          for (auto& p : c) {
            p[point_type::v] = vmin() + (p[point_type::v] - vmin()) * factor;
          }
        }
      }
    }
    else { // rescale u to v
      auto factor = vrange / urange;

      // normalize knotvector
      knotvector_type scaled_knotvector_u;
      for (auto k : knotvector_u()) {
        scaled_knotvector_u.push_back(umin() + (k - umin()) * factor);
      }
      knotvector_u(scaled_knotvector_u.begin(), scaled_knotvector_u.end());

      // normalize trim curves 
      for (auto& tl : _trimloops) {
        for (auto& c : tl) {
          for (auto& p : c) {
            p[point_type::u] = umin() + (p[point_type::u] - umin()) * factor;
          }
        }
      }
    }

  }

  //////////////////////////////////////////////////////////////////////////////
  nurbssurface::trimloop_container const&
  nurbssurface::trimloops() const
  {
      return _trimloops;
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  nurbssurface::print(std::ostream& os) const
  {
     gpucast::math::nurbssurface3d::print(os);

    os << "number of trimloops : " << _trimloops.size() << std::endl;

    os << "trimcurves : " << std::endl;

    for ( curve_container const& loop : _trimloops ) 
    {
      for ( curve_type const& curve : loop )
      {
        curve.print(os);
      }
    }
    
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  nurbssurface::trimtype(bool type)
  {
    _trimtype = type;
  }

  //////////////////////////////////////////////////////////////////////////////
  bool
  nurbssurface::trimtype() const
  {
    return _trimtype;
  }


  //////////////////////////////////////////////////////////////////////////////
  std::ostream& operator<<(std::ostream& os, nurbssurface const& rhs)
  {
    rhs.print(os);
    return os;
  }

} // namespace gpucast

