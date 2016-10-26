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
#include <gpucast/core/singleton.hpp>
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

  _surface_vertex_base_ids.erase(surface);
  _surface_obb_base_ids.erase(surface);
}

////////////////////////////////////////////////////////////////////////////////
beziersurfaceobject::trim_approach_t beziersurfaceobject::trim_approach() const
{
  return _trim_approach;
}

////////////////////////////////////////////////////////////////////////////////
void beziersurfaceobject::trim_approach(beziersurfaceobject::trim_approach_t approach)
{
  if (_trim_approach == approach) {
    return;
  }
  else {
    _trim_approach = approach;

    // overwrite indices with according trimindex information
    for (auto const& surface : _surface_index_map)
    {
      auto const& domain = surface.first->domain();
      auto trim_index = _domain_index_map[domain][_trim_approach];

      // prepare data to update raycasting buffers
      unsigned trim_type_and_approach = uint2x16ToUInt(unsigned short(surface.second.trim_type), unsigned short(_trim_approach));
      float index_as_float = trimdomain_serializer::unsigned_bits_as_float(trim_index);

      for (auto i = surface.second.raycasting_base_index; i != surface.second.raycasting_base_index + surface.second.raycasting_vertex_count; ++i)
      {
        _ray_casting_data.attribute_buffer_2[i] = math::vec4f(index_as_float,
          _ray_casting_data.attribute_buffer_2[i][1],
          bit_cast<unsigned, float>(trim_type_and_approach),
          _ray_casting_data.attribute_buffer_2[i][3]);
      }

      // update tesselation data buffers
      _tesselation_data.patch_data_buffer[surface.second.tesselation_base_index].trim_id = trim_index;
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
  // clear old buffers
  _clearbuffer();

  // apply new parameters
  _subdivision_u                = subdivision_level_u;
  _subdivision_v                = subdivision_level_v;
  _preclassification_resolution = fast_trim_texture_resolution;

  // initialize
  for (surface_ptr const& surface : _surfaces) 
  {
    if (!surface->is_pretrimmable(_preclassification_resolution)) {
      // general preprocessing
      surface->preprocess(_subdivision_u, _subdivision_v);

      // serialize trimming data buffers
      _serialize_trimming_data(surface->domain());

      // serialize raycasting buffers
      _serialize_raycasting_data(surface);

      // serialize tesselation data buffers
      _serialize_adaptive_tesselation_data(surface);
    }
  }

  _is_initialized = true;
}


////////////////////////////////////////////////////////////////////////////////
bool
beziersurfaceobject::initialized () const{
  return _is_initialized;
}

////////////////////////////////////////////////////////////////////////////////
std::unordered_map<beziersurfaceobject::surface_ptr, unsigned> const& beziersurfaceobject::serialized_vertex_base_indices() const{
  return _surface_vertex_base_ids;
}

////////////////////////////////////////////////////////////////////////////////
std::unordered_map<beziersurfaceobject::surface_ptr, unsigned> const& beziersurfaceobject::serialized_obb_base_indices() const{
  return _surface_obb_base_ids;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_double_binary_serialization> beziersurfaceobject::serialized_trimdata_as_double_binary() const {
  return _trimdata_double_binary_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_contour_binary_serialization> beziersurfaceobject::serialized_trimdata_as_contour_binary() const {
  return _trimdata_contour_binary_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_kd_serialization> beziersurfaceobject::serialized_trimdata_as_contour_kd() const {
  return _trimdata_kd_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<trim_loop_list_serialization> beziersurfaceobject::serialized_trimdata_as_contour_loop_list() const {
  return _trimdata_loop_list_serialization;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec3f> const& beziersurfaceobject::serialized_raycasting_data_attrib0() const {
  return _ray_casting_data.attribute_buffer_0;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f>  const& beziersurfaceobject::serialized_raycasting_data_attrib1() const{
  return _ray_casting_data.attribute_buffer_1;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_attrib2() const{
  return _ray_casting_data.attribute_buffer_2;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_attrib3() const{
  return _ray_casting_data.attribute_buffer_3;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_raycasting_data_controlpoints() const{
  return _ray_casting_data.controlpoints;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<unsigned> const& beziersurfaceobject::serialized_raycasting_data_indices() const{
  return _ray_casting_data.index_buffer;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<math::vec4f> const& beziersurfaceobject::serialized_tesselation_domain_buffer() const {
  return _tesselation_data.domain_buffer;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<unsigned> const& beziersurfaceobject::serialized_tesselation_index_buffer() const {
  return _tesselation_data.index_buffer;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<math::vec4f> const& beziersurfaceobject::serialized_tesselation_control_point_buffer() const {
  return _tesselation_data.control_point_buffer;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<beziersurfaceobject::patch_tesselation_data> const& beziersurfaceobject::serialized_tesselation_attribute_data() const {
  return _tesselation_data.patch_data_buffer;
}

////////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f> const& beziersurfaceobject::serialized_tesselation_obbs() const {
  return _tesselation_data.obb_buffer;
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
  _ray_casting_data.attribute_buffer_0.clear();
  _ray_casting_data.attribute_buffer_1.clear();
  _ray_casting_data.attribute_buffer_2.clear();
  _ray_casting_data.attribute_buffer_3.clear();

  // clear index buffer
  _ray_casting_data.index_buffer.clear();

  _surface_vertex_base_ids.clear();
  _surface_obb_base_ids.clear();

  // texture buffer need to be at least one element!
  _ray_casting_data.controlpoints.resize(1);

  // clear trimming data serializations
  _trimdata_double_binary_serialization = std::make_shared<trim_double_binary_serialization>();
  _trimdata_contour_binary_serialization = std::make_shared<trim_contour_binary_serialization>();
  _trimdata_kd_serialization = std::make_shared<trim_kd_serialization>();
  _trimdata_loop_list_serialization = std::make_shared<trim_loop_list_serialization>();

  // clear mappings
  _surface_index_map.clear();
  _domain_index_map.clear();

  // clear tesselation data
  _tesselation_data.domain_buffer.clear();
  _tesselation_data.index_buffer.clear();
  _tesselation_data.control_point_buffer.clear();
  _tesselation_data.patch_data_buffer.clear();
  _tesselation_data.obb_buffer.resize(1);
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
void
beziersurfaceobject::_serialize_raycasting_data(surface_ptr const& surface)
{
  // add control point data into buffer
  std::size_t controlpointdata_index = _add(surface->points());
  _surface_vertex_base_ids.insert({ surface, controlpointdata_index });

  // add convex hull to vbo
  std::size_t chull_index = _add(surface->convexhull());

  // store number of vertices in convex hull
  std::size_t points_in_chull = surface->convexhull().size();

  // store attributes to change trimming
  _surface_index_map[surface].raycasting_base_index = _ray_casting_data.attribute_buffer_2.size();
  _surface_index_map[surface].raycasting_vertex_count = points_in_chull;
  _surface_index_map[surface].trim_type = unsigned(surface->trimtype());
  
  // serialize obb
  std::size_t obb_index = _tesselation_data.obb_buffer.size();
  _surface_obb_base_ids.insert({ surface, obb_index });

  // fill header with index information
  unsigned trim_index = _domain_index_map[surface->domain()][_trim_approach];
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
  std::fill_n(std::back_inserter(_ray_casting_data.attribute_buffer_2), points_in_chull, additional_attrib2);
  std::fill_n(std::back_inserter(_ray_casting_data.attribute_buffer_3), points_in_chull, additional_attrib3);
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
beziersurfaceobject::_add (  gpucast::math::pointmesh2d< gpucast::math::point3d> const& points )
{
  std::size_t index = _ray_casting_data.controlpoints.size();

  for (beziersurface::point_type const& p : points)
  {
    // convert point to hyperspace
    _ray_casting_data.controlpoints.push_back ( gpucast::math::vec4f (float (p[0] * p.weight()),
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
  int offset         = int(_ray_casting_data.attribute_buffer_0.size());
  std::size_t index  = _ray_casting_data.index_buffer.size();

  // copy vertices and start values into client buffer
  std::copy(chull.vertices_begin(), chull.vertices_end(), std::back_inserter(_ray_casting_data.attribute_buffer_0) );
  std::copy(chull.uv_begin(),       chull.uv_end(),       std::back_inserter(_ray_casting_data.attribute_buffer_1) );

  // copy indices into indexbuffer and add buffer offset (convex hull starts by index 0)
  std::transform ( chull.indices_begin(),   
                   chull.indices_end(),   
                   std::back_inserter(_ray_casting_data.index_buffer),
                   [&]( int i ) 
                   { 
                     return i + offset; // add offset to indices
                   } 
                 );

  return index;
}

////////////////////////////////////////////////////////////////////////////////
void beziersurfaceobject::_serialize_obb_data(surface_ptr const& surface)
{
  auto const& obb = surface->obb();

  auto pcenter = math::vec4f(obb.center()[0], obb.center()[1], obb.center()[2], 1.0);
  auto phigh = math::vec4f(obb.high()[0], obb.high()[1], obb.high()[2], 0.0f);
  auto plow = math::vec4f(obb.low()[0], obb.low()[1], obb.low()[2], 0.0f);

  _tesselation_data.obb_buffer.push_back(pcenter);
  _tesselation_data.obb_buffer.push_back(phigh);
  _tesselation_data.obb_buffer.push_back(plow);

  auto orientation = obb.orientation();
  auto inv_orientation = compute_inverse(orientation);

  _tesselation_data.obb_buffer.push_back(math::vec4f(orientation[0][0], orientation[1][0], orientation[2][0], 0.0));
  _tesselation_data.obb_buffer.push_back(math::vec4f(orientation[0][1], orientation[1][1], orientation[2][1], 0.0));
  _tesselation_data.obb_buffer.push_back(math::vec4f(orientation[0][2], orientation[1][2], orientation[2][2], 0.0));
  _tesselation_data.obb_buffer.push_back(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));

  _tesselation_data.obb_buffer.push_back(math::vec4f(inv_orientation[0][0], inv_orientation[1][0], inv_orientation[2][0], 0.0));
  _tesselation_data.obb_buffer.push_back(math::vec4f(inv_orientation[0][1], inv_orientation[1][1], inv_orientation[2][1], 0.0));
  _tesselation_data.obb_buffer.push_back(math::vec4f(inv_orientation[0][2], inv_orientation[1][2], inv_orientation[2][2], 0.0));
  _tesselation_data.obb_buffer.push_back(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));

  auto lbf = math::point3d(plow[0], plow[1], plow[2]);  // left, bottom, front
  auto rbf = math::point3d(phigh[0], plow[1], plow[2]);  // right, bottom, front
  auto rtf = math::point3d(phigh[0], phigh[1], plow[2]);  // right, top, front
  auto ltf = math::point3d(plow[0], phigh[1], plow[2]);  // left, top, front

  auto lbb = math::point3d(plow[0], plow[1], phigh[2]); // left, bottom, back  
  auto rbb = math::point3d(phigh[0], plow[1], phigh[2]); // right, bottom, back  
  auto rtb = math::point3d(phigh[0], phigh[1], phigh[2]); // right, top, back  
  auto ltb = math::point3d(plow[0], phigh[1], phigh[2]); // left, top, back  

  lbf.weight(1.0);
  rbf.weight(1.0);
  rtf.weight(1.0);
  ltf.weight(1.0);

  lbb.weight(1.0);
  rbb.weight(1.0);
  rtb.weight(1.0);
  ltb.weight(1.0);

  _tesselation_data.obb_buffer.push_back(lbf);
  _tesselation_data.obb_buffer.push_back(rbf);
  _tesselation_data.obb_buffer.push_back(rtf);
  _tesselation_data.obb_buffer.push_back(ltf);

  _tesselation_data.obb_buffer.push_back(lbb);
  _tesselation_data.obb_buffer.push_back(rbb);
  _tesselation_data.obb_buffer.push_back(rtb);
  _tesselation_data.obb_buffer.push_back(ltb);
}

////////////////////////////////////////////////////////////////////////////////
void beziersurfaceobject::_serialize_trimming_data(trimdomain_ptr const& domain)
{
  // add trimming information
  trimdomain_serializer_contour_map_binary cmb_serializer;

  std::size_t cmb_index = cmb_serializer.serialize(domain,
    *_trimdata_contour_binary_serialization,
    _preclassification_resolution != 0,
    _preclassification_resolution);

  trimdomain_serializer_double_binary db_serializer;
  std::size_t db_index = db_serializer.serialize(domain,
    *_trimdata_double_binary_serialization,
    _preclassification_resolution != 0,
    _preclassification_resolution);

  trimdomain_serializer_contour_map_kd kd_serializer;
  std::size_t kd_index = kd_serializer.serialize(domain,
    kd_split_strategy::sah,
    *_trimdata_kd_serialization,
    _preclassification_resolution != 0,
    _preclassification_resolution);

  trimdomain_serializer_loop_contour_list loop_serializer;
  std::size_t loop_index = loop_serializer.serialize(domain,
    *_trimdata_loop_list_serialization,
    _preclassification_resolution != 0,
    _preclassification_resolution);

  // store base indices for domain
  _domain_index_map[domain][no_trimming] = 0;
  _domain_index_map[domain][curve_binary_partition] = db_index;
  _domain_index_map[domain][contour_binary_partition] = cmb_index;
  _domain_index_map[domain][contour_kd_partition] = kd_index;
  _domain_index_map[domain][contour_list] = loop_index;
}

////////////////////////////////////////////////////////////////////////////////
void beziersurfaceobject::_serialize_adaptive_tesselation_data(surface_ptr const& surface)
{
  // serialize obb first
  _serialize_obb_data(surface);

  // than serialize rest
  unsigned int patch_id = unsigned int(_tesselation_data.patch_data_buffer.size());
  _surface_index_map[surface].tesselation_base_index = patch_id;

  auto uint_to_float = [](unsigned const & i) { return *((float*)(&i)); };

  auto _p0 = surface->points().begin();
  _tesselation_data.domain_buffer.push_back(math::vec4f((*_p0)[0], (*_p0)[1], (*_p0)[2], uint_to_float(patch_id)));
  _tesselation_data.domain_buffer.push_back(math::vec4f(0.0f, 0.0f, 0.0f, 0.0f));

  auto _p1 = _p0 + surface->points().width() - 1;
  _tesselation_data.domain_buffer.push_back(math::vec4f((*_p1)[0], (*_p1)[1], (*_p1)[2], uint_to_float(patch_id)));
  _tesselation_data.domain_buffer.push_back(math::vec4f(1.0f, 0.0f, 0.0f, 0.0f));

  auto _p2 = surface->points().end() - surface->points().width();
  _tesselation_data.domain_buffer.push_back(math::vec4f((*_p2)[0], (*_p2)[1], (*_p2)[2], uint_to_float(patch_id)));
  _tesselation_data.domain_buffer.push_back(math::vec4f(0.0f, 1.0f, 0.0f, 0.0f));

  auto _p3 = surface->points().end() - 1;
  _tesselation_data.domain_buffer.push_back(math::vec4f((*_p3)[0], (*_p3)[1], (*_p3)[2], uint_to_float(patch_id)));
  _tesselation_data.domain_buffer.push_back(math::vec4f(1.0f, 1.0f, 0.0f, 0.0f));

  auto _v01 = (_p1->as_euclidian()) - (_p0->as_euclidian());
  auto _v13 = (_p3->as_euclidian()) - (_p1->as_euclidian());
  auto _v23 = (_p3->as_euclidian()) - (_p2->as_euclidian());
  auto _v02 = (_p2->as_euclidian()) - (_p0->as_euclidian());

  math::vec4f edge_dist(0.0, 0.0, 0.0, 0.0);

  _tesselation_data.index_buffer.push_back(patch_id * 4 + 0);
  _tesselation_data.index_buffer.push_back(patch_id * 4 + 1);
  _tesselation_data.index_buffer.push_back(patch_id * 4 + 3);
  _tesselation_data.index_buffer.push_back(patch_id * 4 + 2);

  // gather per patch data
  patch_tesselation_data p;
  p.surface_offset = _tesselation_data.control_point_buffer.size();
  p.order_u = surface->order_u();
  p.order_v = surface->order_v();

  p.trim_type = surface->trimtype();
  p.trim_id = _domain_index_map[surface->domain()][_trim_approach];

  auto obb_id = serialized_obb_base_indices().find(surface);
  if (obb_id != serialized_obb_base_indices().end()) {
    p.obb_id = obb_id->second;
  }

  p.nurbs_domain = math::vec4f(surface->bezierdomain().min[0],
    surface->bezierdomain().min[1],
    surface->bezierdomain().max[0],
    surface->bezierdomain().max[1]);
  p.bbox_min = math::vec4f(surface->bbox().min[0],
    surface->bbox().min[1],
    surface->bbox().min[2],
    (float) 0.0f);
  p.bbox_max = math::vec4f(surface->bbox().max[0],
    surface->bbox().max[1],
    surface->bbox().max[2],
    (float) 0.0f);
  p.distance = math::vec4f(std::fabs(edge_dist[0]), std::fabs(edge_dist[1]), std::fabs(edge_dist[2]), std::fabs(edge_dist[3]));

  _tesselation_data.patch_data_buffer.push_back(p);

  // copy patch control points
  int current_size = _tesselation_data.control_point_buffer.size();
  _tesselation_data.control_point_buffer.resize(current_size + surface->points().size());

  auto serialize_homogenous_points = [](gpucast::math::point3d const& p) { auto ph = p.as_homogenous(); return math::vec4f(ph[0], ph[1], ph[2], ph[3]); };
  std::transform(surface->points().begin(), surface->points().end(), _tesselation_data.control_point_buffer.begin() + current_size, serialize_homogenous_points);
}

} // namespace gpucast
