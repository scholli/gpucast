/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziersurfaceobject.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/beziersurfaceobject.hpp"

// header, system

// header, project
#include <gpucast/core/hyperspace_adapter.hpp>
#include <gpucast/core/trimdomain_serializer_double_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_binary.hpp>

namespace gpucast {
  
////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::add(surface_ptr const& surface)
{
  // split up surface into 4 pieces
#if 0
  surface_type bl, tl, br, tr;
  surface->split(bl, tl, br,tr);

  _surfaces.insert( new beziersurface(bl) );
  _surfaces.insert( new beziersurface(tl) );
  _surfaces.insert( new beziersurface(br) );
  _surfaces.insert( new beziersurface(tr) );
#else
  _surfaces.insert(surface);
#endif
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::remove ( surface_ptr const& surface )
{
  _surfaces.erase(surface);
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::print(std::ostream& os) const
{
  os << std::endl << "Bezierobject : " << std::endl;
  std::for_each(_surfaces.begin(), _surfaces.end(), std::bind(&surface_type::print, std::placeholders::_1, std::ref(os)));
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::init(std::size_t subdivision_level_u,
                          std::size_t subdivision_level_v)
{
  _subdivision_u = subdivision_level_u;
  _subdivision_v = subdivision_level_v;

  _clearbuffer();

  std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type> db_domains;
  std::unordered_map<curve_ptr, trimdomain_serializer::address_type> db_curves;

  std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type> cmb_domains;
  std::unordered_map<curve_ptr, trimdomain_serializer::address_type> cmb_curves;
  std::unordered_map<trimdomain::contour_segment_ptr, trimdomain_serializer::address_type> cmb_segments;

  for (surface_ptr const& surface : _surfaces)
  {
    _add(surface, db_domains, db_curves, cmb_domains, cmb_segments, cmb_curves);
  }

  _is_initialized = true;
}


////////////////////////////////////////////////////////////////////////////////
bool
beziersurfaceobject::initialized () const
{
  return _is_initialized;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
beziersurfaceobject::surfaces() const
{
  return _surfaces.size();
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
beziersurfaceobject::trimcurves() const
{
  std::size_t ncurves = 0;
  std::for_each(_surfaces.begin(), _surfaces.end(), [&](surface_ptr const& s) { ncurves += s->trimcurves(); });
  return ncurves;
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::clear()
{
  _surfaces.clear();
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::_clearbuffer()
{
  _attrib0.clear();
  _attrib1.clear();
  _attrib2.clear();
  _attrib3.clear();

  _indices.clear();

  // texture buffer need to be at least one element!
  _controlpoints.resize(1);

  _cmb_partition.resize(1);
  _cmb_contourlist.resize(1);
  _cmb_curvelist.resize(1);
  _cmb_curvedata.resize(1);
  _cmb_pointdata.resize(1);

  _db_partition.resize(1);
  _db_celldata.resize(1);
  _db_curvelist.resize(1);
  _db_curvedata.resize(1);
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::merge(beziersurfaceobject const& other)
{
  std::copy(other.begin(), other.end(), std::inserter(_surfaces, _surfaces.end()));

  _is_initialized = false;
  _clearbuffer();
}


////////////////////////////////////////////////////////////////////////////////
std::map<int, unsigned>
beziersurfaceobject::order_surfaces() const
{
  std::map<int, unsigned> order_surface_map;
  for (surface_ptr const& surface : _surfaces)
  {
    ++order_surface_map[int(std::max(surface->order_u(), surface->order_v()))];
  }
  return order_surface_map;
}

////////////////////////////////////////////////////////////////////////////////
std::map<int, unsigned>
beziersurfaceobject::order_trimcurves() const
{
  std::map<int, unsigned> order_trimcurve_map;
  for (surface_ptr const& surface : _surfaces)
  {
    for (auto curve : surface->domain()->curves())
    {
      ++order_trimcurve_map[int(curve->order())];
    }
  }
  return order_trimcurve_map;
}

////////////////////////////////////////////////////////////////////////////////
beziersurfaceobject::bbox_t
beziersurfaceobject::bbox() const
{
  double xmin =  std::numeric_limits<double>::max();
  double xmax = -std::numeric_limits<double>::max();
  double ymin =  std::numeric_limits<double>::max();
  double ymax = -std::numeric_limits<double>::max();
  double zmin =  std::numeric_limits<double>::max();
  double zmax = -std::numeric_limits<double>::max();

  for (surface_ptr const& surface : _surfaces)
  {
    bbox_t tmp = surface->bbox();
    xmax = std::max(xmax, tmp.max[0]);
    ymax = std::max(ymax, tmp.max[1]);
    zmax = std::max(zmax, tmp.max[2]);
    xmin = std::min(xmin, tmp.min[0]);
    ymin = std::min(ymin, tmp.min[1]);
    zmin = std::min(zmin, tmp.min[2]);
  }

  return bbox_t( gpucast::math::point3d(xmin, ymin, zmin),
                 gpucast::math::point3d(xmax, ymax, zmax));
}

////////////////////////////////////////////////////////////////////////////////
std::size_t             
beziersurfaceobject::size () const
{
  return _surfaces.size();
}

////////////////////////////////////////////////////////////////////////////////
beziersurfaceobject::const_surface_iterator  
beziersurfaceobject::begin () const
{
  return _surfaces.begin();
}

////////////////////////////////////////////////////////////////////////////////
beziersurfaceobject::const_surface_iterator  
beziersurfaceobject::end () const
{
  return _surfaces.end();
}

////////////////////////////////////////////////////////////////////////////////
std::size_t
beziersurfaceobject::_add ( surface_ptr const& surface,
                            std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type>&                  db_domains,
                            std::unordered_map<curve_ptr, trimdomain_serializer::address_type>&                       db_curves,
                            std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type>&                  cmb_domains,
                            std::unordered_map<trimdomain::contour_segment_ptr, trimdomain_serializer::address_type>& cmb_segments, 
                            std::unordered_map<curve_ptr, trimdomain_serializer::address_type>&                       cmb_curves )
{
  // preprocess surface if it was modified or not initialized
  surface->preprocess(_subdivision_u, _subdivision_v);

  // add control point data into buffer
  std::size_t controlpointdata_index  = _add (surface->points());

  // add trimming information
  trimdomain_serializer_contour_map_binary cmb_serializer;
  std::size_t cmb_index = cmb_serializer.serialize ( surface->domain(), 
                                                     cmb_domains, 
                                                     cmb_curves, 
                                                     cmb_segments, 
                                                     _cmb_partition, 
                                                     _cmb_contourlist, 
                                                     _cmb_curvelist, 
                                                     _cmb_curvedata, 
                                                     _cmb_pointdata );

  trimdomain_serializer_double_binary db_serializer;
  std::size_t db_index = db_serializer.serialize ( surface->domain(), 
                                                   db_domains, 
                                                   db_curves, 
                                                   _db_partition, 
                                                   _db_celldata, 
                                                   _db_curvelist, 
                                                   _db_curvedata );


  // add convex hull to vbo
  std::size_t chull_index = _add(surface->convexhull());

  // store number of vertices in convex hull
  std::size_t points_in_chull = surface->convexhull().size();

  gpucast::math::vec4f additional_attrib2(cmb_serializer.unsigned_bits_as_float(explicit_type_conversion<std::size_t, unsigned> (db_index) ),
                                        cmb_serializer.unsigned_bits_as_float(explicit_type_conversion<std::size_t, unsigned>(controlpointdata_index)),
                                        cmb_serializer.unsigned_bits_as_float( surface->trimtype() ),
                                        cmb_serializer.unsigned_bits_as_float(explicit_type_conversion<std::size_t, unsigned>(cmb_index)));

  gpucast::math::vec4f additional_attrib3 ( float(surface->bezierdomain().min[trimdomain::point_type::u]),
                                          float(surface->bezierdomain().max[trimdomain::point_type::u]),
                                          float(surface->bezierdomain().min[trimdomain::point_type::v]),
                                          float(surface->bezierdomain().max[trimdomain::point_type::v]) );

  // blow attribute arrays to fill them with additional attributes
  std::fill_n(std::back_inserter(_attrib2), points_in_chull, additional_attrib2);
  std::fill_n(std::back_inserter(_attrib3), points_in_chull, additional_attrib3);

  return chull_index;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
beziersurfaceobject::_add (  gpucast::math::pointmesh2d< gpucast::math::point3d> const& points )
{
  std::size_t index = _controlpoints.size();

  for (beziersurface::point_type const& p : points)
  {
    // convert point to hyperspace
    _controlpoints.push_back ( gpucast::math::vec4f (float (p[0] * p.weight()),
                                            float (p[1] * p.weight()),
                                            float (p[2] * p.weight()),
                                            float (p.weight())));
  }

  return index;
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
beziersurfaceobject::_add ( convex_hull const& chull )
{
  int offset         = int(_attrib0.size());
  std::size_t index  = _indices.size();

  // copy vertices and start values into client buffer
  std::copy(chull.vertices_begin(), chull.vertices_end(), std::back_inserter(_attrib0) );
  std::copy(chull.uv_begin(),       chull.uv_end(),       std::back_inserter(_attrib1) );

  // copy indices into indexbuffer and add buffer offset (convex hull starts by index 0)
  std::transform ( chull.indices_begin(),   
                   chull.indices_end(),   
                   std::back_inserter(_indices),  
                   [&]( int i ) 
                   { 
                     return i + offset; // add offset to indices
                   } 
                 );

  return index;
}

} // namespace gpucast
