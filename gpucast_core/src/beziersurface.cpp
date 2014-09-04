/********************************************************************************
*
* Copyright (C) 2008 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziersurface.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#include "gpucast/core/beziersurface.hpp"

// header, system

// header, project
#include <gpucast/core/uvgenerator.hpp>
#include <gpucast/math/vec3.hpp>

using namespace gpucast::math;

namespace gpucast {

  //////////////////////////////////////////////////////////////////////////////
  beziersurface::beziersurface ( )
    : beziersurface3d ( ),
      _chull          ( ),
      _trimdomain     ( ),
      _bezierdomain   ( curve_point_type (0, 0), curve_point_type (1, 1) )
  {}


  //////////////////////////////////////////////////////////////////////////////
  beziersurface::beziersurface  ( beziersurface3d const& untrimmed_surface )
    : beziersurface3d ( untrimmed_surface ),
      _chull          ( ),
      _trimdomain     ( ),
      _bezierdomain   ( curve_point_type (0, 0), curve_point_type (1, 1) )
  {}


  //////////////////////////////////////////////////////////////////////////////
  beziersurface::beziersurface ( beziersurface const& bs )
    : beziersurface3d           ( bs ),
      _chull                    ( bs._chull ),
      _trimdomain               ( bs._trimdomain ),
      _bezierdomain             ( bs._bezierdomain )
  {}


  //////////////////////////////////////////////////////////////////////////////
  beziersurface::~beziersurface()
  {}


  //////////////////////////////////////////////////////////////////////////////
  void
  beziersurface::swap(beziersurface& other)
  {
    beziersurface3d::swap(other);

    std::swap(_chull,         other._chull);
    std::swap(_trimdomain,    other._trimdomain);
    std::swap(_bezierdomain,  other._bezierdomain);
  }


  //////////////////////////////////////////////////////////////////////////////
  beziersurface&
  beziersurface::operator=(beziersurface const& cpy)
  {
    beziersurface tmp(cpy);
    swap(tmp);
    return *this;
  }


  //////////////////////////////////////////////////////////////////////////////
  /*void
  beziersurface::add (curve_type const& trim)
  {
    _trimdomain.add(trim);
  }*/


  //////////////////////////////////////////////////////////////////////////////
  /* virtual */ void     
  beziersurface::split ( beziersurface& bl,
                         beziersurface& tl,
                         beziersurface& br,
                         beziersurface& tr ) const
  {
    base_type::split(bl, tl, br, tr);

    bl._trimdomain    = _trimdomain;
    tl._trimdomain    = _trimdomain;
    br._trimdomain    = _trimdomain;
    tr._trimdomain    = _trimdomain;

    trimdomain::bbox_type bdom = _bezierdomain;
    
    bl.bezierdomain ( trimdomain::bbox_type( curve_point_type ( _bezierdomain.min[curve_point_type::u],      _bezierdomain.min[curve_point_type::v]      ), curve_point_type (_bezierdomain.center()[curve_point_type::u], _bezierdomain.center()[curve_point_type::v]))); 
    tl.bezierdomain ( trimdomain::bbox_type( curve_point_type ( _bezierdomain.min[curve_point_type::u],      _bezierdomain.center()[curve_point_type::v] ), curve_point_type (_bezierdomain.center()[curve_point_type::u], _bezierdomain.max[curve_point_type::v]     ))); 
    br.bezierdomain ( trimdomain::bbox_type( curve_point_type ( _bezierdomain.center()[curve_point_type::u], _bezierdomain.min[curve_point_type::v]      ), curve_point_type (_bezierdomain.max[curve_point_type::u],      _bezierdomain.center()[curve_point_type::v]))); 
    tr.bezierdomain ( trimdomain::bbox_type( curve_point_type ( _bezierdomain.center()[curve_point_type::u], _bezierdomain.center()[curve_point_type::v] ), curve_point_type (_bezierdomain.max[curve_point_type::u],      _bezierdomain.max[curve_point_type::v]     ))); 
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  beziersurface::trimtype(bool t)
  {
    _trimdomain->type(t);
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  beziersurface::nurbsdomain(trimdomain::bbox_type const& n)
  {
    _trimdomain->nurbsdomain(n);
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  beziersurface::bezierdomain(trimdomain::bbox_type const& b)
  {
    _bezierdomain = b;
  }


  //////////////////////////////////////////////////////////////////////////////
  trimdomain::bbox_type const& 
  beziersurface::bezierdomain() const
  {
    return _bezierdomain;
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  beziersurface::preprocess ( std::size_t   subdiv_u, 
                              std::size_t   subdiv_v )
  {
    // clear old convex hull
    _chull.clear();

    // elevate if necessary
    if (_degree_u < 2) {
      elevate_u();
    }

    if (_degree_v < 2) {
      elevate_v();
    }

    // compute parameter domain partition if necessary
    //_trimdomain.partition();

    // generate convex hull according to subdivision parameter
    pointmesh2d<point_type> cp_orig(_points.begin(), _points.end(), order_u(), order_v());
    pointmesh2d<point_type> cp_u;
    pointmesh2d<point_type> cp_uv;

    trimdomain::bbox_type domainsize = _bezierdomain;
    double umin = _bezierdomain.min[point_type::u];
    double umax = _bezierdomain.max[point_type::u];
    double vmin = _bezierdomain.min[point_type::v];
    double vmax = _bezierdomain.max[point_type::v];

    value_type range_u = umax - umin;
    value_type range_v = vmax - vmin;
    value_type step_u = range_u / value_type(subdiv_u+1);
    value_type step_v = range_v / value_type(subdiv_v+1);

    // for each row of control polygon
    for (std::size_t r = 0; r <= _degree_v; ++r)
    {
      // create knot vector
      std::vector<value_type>    hlp1(_degree_u+1, umin);
      std::vector<value_type>    hlp2(_degree_u+1, umax);

      std::multiset<value_type>  kv_u(hlp1.begin(), hlp1.end());

      kv_u.insert(hlp2.begin(), hlp2.end());

      std::vector<point_type> row = cp_orig.row(r);

      for (std::size_t i = 1; i <= subdiv_u; ++i)
      {
        converter3d conv;
        conv.knot_insertion(row, kv_u, _degree_u + 1, value_type(umin) + value_type(i) * value_type(step_u));
      }

      cp_u.add_row(row.begin(), row.end());
    }

    cp_u.transpose();

    // for each row of transposed control polygon
    for (std::size_t r = 0; r < cp_u.height(); ++r)
    {
      // create knot vector
      std::vector<value_type> hlp1(_degree_v+1, vmin);
      std::vector<value_type> hlp2(_degree_v+1, vmax);
      std::multiset<value_type> kv_v(hlp1.begin(), hlp1.end());
      kv_v.insert(hlp2.begin(), hlp2.end());

      std::vector<point_type> row = cp_u.row(r);

      for (std::size_t i = 1; i <= subdiv_v; ++i)
      {
        converter3d conv;
        conv.knot_insertion(row, kv_v,  _degree_v + 1,  value_type(vmin) + value_type(i) * value_type(step_v));
      }

      cp_uv.add_row(row.begin(), row.end());
    }

    cp_uv.transpose();

    std::vector<gpucast::math::vec4f> uv(cp_uv.width() * cp_uv.height());
    //    std::generate(uv.begin(), uv.end(), gen_uv(cp_uv.width(), cp_uv.height(),
    //					       min_u_, max_u_, min_v_, max_v_));

    // for bezier patches always 0.0 - 1.0 -> min_u, max_u, min_v, max_v are only for trimming ...
    // ... as they transform the parameter space into the original b-spline space
    std::generate(uv.begin(), uv.end(), uvgenerator<gpucast::math::vec4f> ( cp_uv.width(), 
                                                                   cp_uv.height(), 
                                                                   float(0), 
                                                                   float(1), 
                                                                   float(0), 
                                                                   float(1), 
                                                                   float(order_u()), 
                                                                   float(order_v())));

    std::vector<gpucast::math::vec4f> attrib(uv.size());

    pointmesh2d<gpucast::math::vec4f> cp_color (uv.begin(),  uv.end(), cp_uv.width(), cp_uv.height());
    pointmesh2d<gpucast::math::vec4f> cp_attrib(uv.begin(),  uv.end(), cp_uv.width(), cp_uv.height());

    // generate chulls for each subdivision and merge them
    for (std::size_t v = 0; v <= subdiv_v; ++v)
    {
      for (std::size_t u = 0; u <= subdiv_u; ++u)
      {
        // copy control points of submesh
        pointmesh2d<point3d>      subpat     = cp_uv.subpatch    (u * _degree_u, (u+1)*_degree_u, v * _degree_v, (v+1)*_degree_v);

        // copy 4-double texture coordinate information from submesh
        pointmesh2d<gpucast::math::vec4f>  subpat_col = cp_color.subpatch (u * _degree_u, (u+1)*_degree_u, v * _degree_v, (v+1)*_degree_v);

        // discard rational component of control points for convex hull generation
        std::vector<gpucast::math::vec3d> tmp;
        std::transform(subpat.begin(), subpat.end(), std::inserter(tmp, tmp.end()), gpucast::math::rational_to_euclid3d<point3d>());

        // generate chull for subpatch
        convex_hull tmp_ch;
        tmp_ch.set(tmp.begin(), tmp.end(), subpat_col.begin(), subpat_col.end());

        // merge chull with complete chull
        _chull.merge(tmp_ch);
      }
    }
  }


  //////////////////////////////////////////////////////////////////////////////
  bool
  beziersurface::trimtype() const
  {
    return _trimdomain->type();
  }


  //////////////////////////////////////////////////////////////////////////////
  std::size_t
  beziersurface::trimcurves() const
  {
    return _trimdomain->size();
  }


  //////////////////////////////////////////////////////////////////////////////
  beziersurface::trimdomain_ptr const&
  beziersurface::domain() const
  {
    return _trimdomain;
  }


  //////////////////////////////////////////////////////////////////////////////
  void                   
  beziersurface::domain ( trimdomain_ptr const& domain )
  {
    _trimdomain = domain;
  }


  //////////////////////////////////////////////////////////////////////////////
  convex_hull const&
  beziersurface::convexhull() const
  {
    return _chull;
  }


  //////////////////////////////////////////////////////////////////////////////
  beziersurface::mesh_type const&
  beziersurface::points() const
  {
    return _points;
  }


  //////////////////////////////////////////////////////////////////////////////
  void
  beziersurface::print(std::ostream& os) const
  {
    beziersurface3d::print(os);

    os << "rational trimming curves : " << _trimdomain->size() << std::endl;

    _trimdomain->print(os);
  }



} // namespace gpucast
