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
  void trimdomain::normalize()
  {
    double size_u = _nurbsdomain.max[0] - _nurbsdomain.min[0];
    double size_v = _nurbsdomain.max[1] - _nurbsdomain.min[1];

    for (contour_type const& loop : _trimloops) {
      for (auto c : loop) {
        for (auto p = c->begin(); p != c->end(); ++p) {
          (*p)[0] = ((*p)[0] - _nurbsdomain.min[0]) / size_u;
          (*p)[1] = ((*p)[1] - _nurbsdomain.min[1]) / size_v;
        }
      }
    }
    _nurbsdomain.min = point2d{ 0,0 };
    _nurbsdomain.max = point2d{ 1,1 };
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
  grid<trimdomain::value_type> trimdomain::signed_distance_field(unsigned resolution) const
  {
    if (_signed_distance_fields.count(resolution)) {
      return _signed_distance_fields.at(resolution);
    }
    else {

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

      grid<value_type> result(resolution, resolution);
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

      _signed_distance_fields.insert(std::make_pair(resolution, result));
      return result;
    }
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
  grid<unsigned char> trimdomain::signed_distance_pre_classification(unsigned resolution) const
  {
    grid<unsigned char> result(resolution, resolution);

    auto sdf = signed_distance_field(resolution);
    auto texel_diameter = nurbsdomain().size() / resolution;
    auto texel_radius = texel_diameter.abs() / 2;

    std::transform(sdf.begin(), sdf.end(), result.begin(), [texel_radius](double v) {
      if (v < -texel_radius)
        return trimmed; // classified outside
      if (v > texel_radius)
        return untrimmed; // classified inside
      else
        return unknown; // not classified
    });

    return result;
  }

  /////////////////////////////////////////////////////////////////////////////
  grid<unsigned char> trimdomain::pre_classification(unsigned resolution) const
  {
    grid<unsigned char> result(resolution, resolution);

    gpucast::math::vec2d texel_size_uv = nurbsdomain().size() / resolution;

    // inline lambda for texel classification
    auto classify_texel = [&](gpucast::math::bbox2d const& texel) -> int {
      int curves_in_positive_u = 0;
      for (auto const& l : loops()) {
        for (auto const& c : l.curves()) {
          // if curve bounding box overlaps texel -> no easy early classification possible
          auto curve_bbox = c->bbox_simple(); 

          if (texel.overlap(curve_bbox)) {
            return unknown;
          }

          // start ray from texel center
          auto origin = point2d(texel.max[0], (texel.min[1] + texel.max[1])/2.0 );
          intervald vrange{ c->front()[1] , c->back()[1], included, included };

          bool origin_is_v_interval = vrange.in(origin[1]);
          bool origin_is_left_of_curve = origin[0] < curve_bbox.min[0];

          if (origin_is_v_interval && origin_is_left_of_curve) {
            curves_in_positive_u++;
          }
        }
      }

      // if outer part is trimmed
      bool odd_number_intersections = curves_in_positive_u % 2 == 1;
      if (type() == true) {
        // if odd number of intersections        
        return odd_number_intersections ? untrimmed : trimmed;
      }
      else {
        return odd_number_intersections ? trimmed : untrimmed;
      }
      return unknown;
    };

    // iterate over all texels
    for (int v = 0; v != resolution; ++v) {
      for (int u = 0; u != resolution; ++u) {
        gpucast::math::bbox2d texel = { point2d{ nurbsdomain().min[0] + u * texel_size_uv[0],
                                                 nurbsdomain().min[1] + v * texel_size_uv[1] },
                                        point2d{ nurbsdomain().min[0] + (u + 1) * texel_size_uv[0],
                                                 nurbsdomain().min[1] + (v+1) * texel_size_uv[1] } };
        auto c = classify_texel(texel);
        result(u, v) = c;
      }  
    }
    return result;
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
