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
#include <gpucast/core/trimdomain_serializer_contour_map_kd.hpp>
#include <gpucast/core/trimdomain_serializer_loop_contour_list.hpp>

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::add(surface_ptr const& surface) 
{
  _surfaces.insert(surface);
}


////////////////////////////////////////////////////////////////////////////////
void
beziersurfaceobject::remove ( surface_ptr const& surface ) 
{
  _surfaces.erase(surface);
}

////////////////////////////////////////////////////////////////////////////////
beziersurfaceobject::trim_approach_t beziersurfaceobject::trim_approach() const
{
  return _trim_approach;
}

////////////////////////////////////////////////////////////////////////////////
void beziersurfaceobject::trim_approach(beziersurfaceobject::trim_approach_t approach)
{
  _trim_approach = approach;
  trimdomain_serializer serializer; 

  // overwrite indices with according trimindex information
  for (auto const& surface : _surface_index_map) 
  {
    auto const& domain = surface.first->domain();
    auto indices = _domain_index_map[domain];

    unsigned trim_index = 0;
   
    switch (_trim_approach) {
      case no_trimming:
        trim_index = 0; break;
      case curve_binary_partition:
        trim_index = explicit_type_conversion<std::size_t, unsigned>(indices.double_binary_index); break;
      case contour_binary_partition:
        trim_index = explicit_type_conversion<std::size_t, unsigned>(indices.contour_binary_index); break;
      case contour_kd_partition:
        trim_index = explicit_type_conversion<std::size_t, unsigned>(indices.contour_kd_index); break;
      case contour_list:
        trim_index = explicit_type_conversion<std::size_t, unsigned>(indices.loop_list_index); break;
      default:
        trim_index = 0;
    };

    unsigned trim_type_and_approach = uint2x16ToUInt(unsigned short(surface.second.trim_type), unsigned short(_trim_approach));
    float index_as_float = serializer.unsigned_bits_as_float(trim_index);

    for (std::size_t i = surface.second.base_index; i != surface.second.base_index + surface.second.count; ++i) {
      _attrib2[i] = math::vec4f(index_as_float, 
                                _attrib2[i][1], 
                                bit_cast<unsigned, float>(trim_type_and_approach), 
                                _attrib2[i][3]);
    }

  }
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
beziersurfaceobject::init(unsigned subdivision_level_u,
                          unsigned subdivision_level_v, 
                          unsigned fast_trim_texture_resolution) 
{
  _subdivision_u = subdivision_level_u;
  _subdivision_v = subdivision_level_v;

  _clearbuffer();

  for (surface_ptr const& surface : _surfaces) 
  {
    _add(surface, fast_trim_texture_resolution);
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
std::shared_ptr<trim_double_binary_serialization> beziersurfaceobject::serialized_trimdata_as_double_binary() const
{
  return _trimdata_double_binary_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_contour_binary_serialization> beziersurfaceobject::serialized_trimdata_as_contour_binary() const
{
  return _trimdata_contour_binary_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_kd_serialization> beziersurfaceobject::serialized_trimdata_as_contour_kd() const
{
  return _trimdata_kd_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_loop_list_serialization> beziersurfaceobject::serialized_trimdata_as_contour_loop_list() const
{
  return _trimdata_loop_list_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec3f> const& beziersurfaceobject::serialized_raycasting_data_attrib0() const
{
  return _attrib0;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f>  const& beziersurfaceobject::serialized_raycasting_data_attrib1() const
{
  return _attrib1;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_attrib2() const
{
  return _attrib2;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_attrib3() const
{
  return _attrib3;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_controlpoints() const
{
  return _controlpoints;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_obbs() const
{
  return _obbs;
}


////////////////////////////////////////////////////////////////////////////////
std::vector<unsigned> const& beziersurfaceobject::serialized_raycasting_data_indices() const
{
  return _indices;
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
  // clear vertex attribute buffers
  _attrib0.clear();
  _attrib1.clear();
  _attrib2.clear();
  _attrib3.clear();

  // clear index buffer
  _indices.clear();

  // texture buffer need to be at least one element!
  _controlpoints.resize(1);
  _obbs.resize(1);

  // clear trimming data serializations
  _trimdata_double_binary_serialization = std::make_shared<trim_double_binary_serialization>();
  _trimdata_contour_binary_serialization = std::make_shared<trim_contour_binary_serialization>();
  _trimdata_kd_serialization = std::make_shared<trim_kd_serialization>();
  _trimdata_loop_list_serialization = std::make_shared<trim_loop_list_serialization>();

  // clear mappings
  _surface_index_map.clear();
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
beziersurfaceobject::_add(surface_ptr surface, unsigned fast_trim_texture_resolution)
{
  // preprocess surface if it was modified or not initialized
  surface->preprocess(_subdivision_u, _subdivision_v);

  // add control point data into buffer
  std::size_t controlpointdata_index  = _add (surface->points());

  // add trimming information
  trimdomain_serializer_contour_map_binary cmb_serializer;
  std::size_t cmb_index = cmb_serializer.serialize ( surface->domain(), 
                                                     *_trimdata_contour_binary_serialization,
                                                     fast_trim_texture_resolution != 0,
                                                     fast_trim_texture_resolution);

  trimdomain_serializer_double_binary db_serializer;
  std::size_t db_index = db_serializer.serialize ( surface->domain(), 
                                                   *_trimdata_double_binary_serialization,
                                                   fast_trim_texture_resolution != 0,
                                                   fast_trim_texture_resolution);

  trimdomain_serializer_contour_map_kd kd_serializer;
  std::size_t kd_index = kd_serializer.serialize(surface->domain(),
                                                 kd_split_strategy::sah,
                                                 *_trimdata_kd_serialization,
                                                 fast_trim_texture_resolution != 0,
                                                 fast_trim_texture_resolution);

  trimdomain_serializer_loop_contour_list loop_serializer;
  std::size_t loop_index = loop_serializer.serialize(surface->domain(),
                                                     *_trimdata_loop_list_serialization,
                                                     fast_trim_texture_resolution != 0,
                                                     fast_trim_texture_resolution);

  // add convex hull to vbo
  std::size_t chull_index = _add(surface->convexhull());

  // store number of vertices in convex hull
  std::size_t points_in_chull = surface->convexhull().size();

  // store attributes to change trimming
  _surface_index_map[surface] = { _attrib2.size(), points_in_chull, unsigned(surface->trimtype()) };
  _domain_index_map[surface->domain()] = { db_index, cmb_index, kd_index, loop_index };

  // serialize obb
  std::size_t obb_index = _obbs.size();

  auto const& obb = surface->obb();

  auto pcenter = math::vec4f(obb.center()[0], obb.center()[1], obb.center()[2], 1.0);
  auto phigh   = math::vec4f(obb.high()[0], obb.high()[1], obb.high()[2], 0.0f);
  auto plow    = math::vec4f(obb.low()[0], obb.low()[1], obb.low()[2], 0.0f);

  _obbs.push_back(pcenter);
  _obbs.push_back(phigh);
  _obbs.push_back(plow);

  auto orientation = surface->obb().orientation();
  auto inv_orientation = compute_inverse(orientation);

  _obbs.push_back(math::vec4f(orientation[0][0], orientation[0][1], orientation[0][2], 0.0));
  _obbs.push_back(math::vec4f(orientation[1][0], orientation[1][1], orientation[1][2], 0.0));
  _obbs.push_back(math::vec4f(orientation[2][0], orientation[2][1], orientation[2][2], 0.0));
  _obbs.push_back(math::vec4f(0.0f, 0.0f, 0.0f, 0.0f));

  _obbs.push_back(math::vec4f(inv_orientation[0][0], inv_orientation[0][1], inv_orientation[0][2], 0.0));
  _obbs.push_back(math::vec4f(inv_orientation[1][0], inv_orientation[1][1], inv_orientation[1][2], 0.0));
  _obbs.push_back(math::vec4f(inv_orientation[2][0], inv_orientation[2][1], inv_orientation[2][2], 0.0));
  _obbs.push_back(math::vec4f(0.0f, 0.0f, 0.0f, 0.0f));

  auto lbf = orientation * math::point3d(plow[0], plow[1], plow[2]) + obb.center();  // left, bottom, front
  auto rbf = orientation * math::point3d(phigh[0], plow[1], plow[2]) + obb.center();  // right, bottom, front
  auto rtf = orientation * math::point3d(phigh[0], phigh[1], plow[2]) + obb.center();  // right, top, front
  auto ltf = orientation * math::point3d(plow[0], plow[1], plow[2]) + obb.center();  // left, top, front
  auto lbb = orientation * math::point3d(plow[0], plow[1], phigh[2]) + obb.center(); // left, bottom, back  
  auto rbb = orientation * math::point3d(phigh[0], plow[1], phigh[2]) + obb.center(); // right, bottom, back  
  auto rtb = orientation * math::point3d(phigh[0], phigh[1], phigh[2]) + obb.center(); // right, top, back  
  auto ltb = orientation * math::point3d(plow[0], phigh[1], phigh[2]) + obb.center(); // left, top, back  

  lbf.weight(1.0);
  rbf.weight(1.0);
  rtf.weight(1.0);
  ltf.weight(1.0);
  lbb.weight(1.0);
  rbb.weight(1.0);
  rtb.weight(1.0);
  ltb.weight(1.0);

  _obbs.push_back(lbf);
  _obbs.push_back(rbf);
  _obbs.push_back(rtf);
  _obbs.push_back(ltf);
  _obbs.push_back(lbb);
  _obbs.push_back(rbb);
  _obbs.push_back(rtb);
  _obbs.push_back(ltb);

  // fill header with index information
  unsigned trim_index = 0;
  switch (_trim_approach) 
  {
    case no_trimming :    
      trim_index = 0; break;
    case curve_binary_partition:
      trim_index = db_index; break;
    case contour_binary_partition:
      trim_index = cmb_index; break;
    case contour_kd_partition:
      trim_index = kd_index; break;
    case contour_list:
      trim_index = loop_index; break;
    default: 
      trim_index = 0;
  };

  unsigned trim_type_and_approach = uint2x16ToUInt(unsigned short(surface->trimtype()), unsigned short(_trim_approach));

  gpucast::math::vec4f additional_attrib2(bit_cast<unsigned, float>(explicit_type_conversion<std::size_t, unsigned>(trim_index)),
                                          bit_cast<unsigned, float>(explicit_type_conversion<std::size_t, unsigned>(controlpointdata_index)),
                                          bit_cast<unsigned, float>(trim_type_and_approach),
                                          bit_cast<unsigned, float>(explicit_type_conversion<std::size_t, unsigned>(obb_index))
                                          );

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
