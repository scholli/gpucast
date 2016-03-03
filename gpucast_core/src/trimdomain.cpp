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
#include <thread>

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
    if (_trimloops.empty()) {
      return false;
    }
    else {
      return _type;
    }
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
  std::vector<trimdomain::value_type> trimdomain::signed_distance_field(unsigned resolution) const
  {
    std::size_t const worker_threads = 16;

    unsigned res_u = resolution;
    unsigned res_v = resolution;

    auto offset_u = (_nurbsdomain.max[point_type::u] - _nurbsdomain.min[point_type::u]) / res_u;
    auto offset_v = (_nurbsdomain.max[point_type::v] - _nurbsdomain.min[point_type::v]) / res_v;

    auto min_u = _nurbsdomain.min[point_type::u];
    auto min_v = _nurbsdomain.min[point_type::v];

    std::vector<bbox_type> texels;

    // create jobs
    for (unsigned v = 0; v < res_v; ++v)
    {
      for (unsigned u = 0; u < res_u; ++u)
      {
        auto bbox_min = point_type(min_u + u*offset_u, min_v + v*offset_v);
        auto bbox_max = point_type(min_u + (u + 1)*offset_u, min_v + (v + 1)*offset_v);

        bbox_type texel(bbox_min, bbox_max);
        texels.push_back(texel);
      }
    }

    std::vector<value_type> result(texels.size());
    std::vector<std::thread> threads(worker_threads);

    auto worker = [&result, &texels, worker_threads](trimdomain const* domain, std::size_t id) {
      while (id < result.size()) {
        result[id] = domain->signed_distance(texels[id].center());
        id += worker_threads;
      }
    };

    std::size_t thread_id = 0;
    for (auto& t : threads) {
      t = std::thread(std::bind(worker, this, thread_id++));
    }
    
    for (auto& t : threads) {
      t.join();
    }

    return result;
  }

  /////////////////////////////////////////////////////////////////////////////
  trimdomain::value_type trimdomain::signed_distance(point_type const& p) const
  {
    unsigned intersections = 0;
    value_type minimal_distance = std::numeric_limits<value_type>::max();

    for (auto const& loop : loops()) {
      intersections += loop.is_inside(p);
      for (auto const& curve : loop.curves()) {
        if (curve->bbox_simple().distance(p) < minimal_distance)
        {
          minimal_distance = std::min(curve->closest_distance(p), minimal_distance);
        }
      }
    }

    if (intersections % 2) {
      return minimal_distance;
    }
    else {
      return -minimal_distance;
    }
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
