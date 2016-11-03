/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziersurfaceobject.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_BEZIERSURFACEOBJECT_HPP
#define GPUCAST_CORE_BEZIERSURFACEOBJECT_HPP

// header, system

// header, external
#include <gpucast/math/axis_aligned_boundingbox.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/util.hpp>
#include <gpucast/core/beziersurface.hpp>
#include <gpucast/core/surface_renderer.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

struct trim_double_binary_serialization;
struct trim_contour_binary_serialization;
struct trim_kd_serialization;
struct trim_loop_list_serialization;
class surface_renderer;

class GPUCAST_CORE beziersurfaceobject
{
public : // friends

  friend class surface_renderer_gl;

  static const unsigned trim_preclassification_default_resolution = 8;
  static const unsigned default_initial_subdivision = 0;

public : // enums, typedefs

  enum trim_approach_t {
    no_trimming              = 0,
    curve_binary_partition   = 1,
    contour_binary_partition = 2,
    contour_kd_partition     = 3,
    contour_list             = 4,
    count                    = 5
  };

  struct patch_tesselation_data
  {
    unsigned surface_offset;
    unsigned char order_u;
    unsigned char order_v;
    unsigned short trim_type;
    unsigned trim_id;
    unsigned obb_id;

    math::vec4f nurbs_domain;
    math::vec4f bbox_min;
    math::vec4f bbox_max;
    math::vec4f distance;
  };

  struct GPUCAST_CORE memory_usage {
    // unavoidable data
    std::size_t surface_control_point_data = 0;
    std::size_t trimcurve_control_point_data = 0;

    // renderable data
    std::size_t vertex_array_raycasting = 0;
    std::size_t vertex_array_tesselation = 0;

    // trim domain partition summed up per approach
    std::size_t domain_partition_kd_tree = 0;
    std::size_t domain_partition_contour_binary = 0;
    std::size_t domain_partition_double_binary = 0;
    std::size_t domain_partition_loops = 0;

    memory_usage& operator+=(memory_usage const& rhs);
  };

  typedef beziersurface::value_type         value_type;
  typedef beziersurface                     surface_type;
  typedef std::shared_ptr<surface_type>     surface_ptr;
  typedef std::set<surface_ptr>             surface_container;
  typedef surface_container::iterator       surface_iterator;
  typedef surface_container::const_iterator const_surface_iterator;

  typedef beziersurface::curve_type         curve_type;
  typedef std::shared_ptr<curve_type>       curve_ptr;

  typedef surface_type::trimdomain_ptr      trimdomain_ptr;

  typedef gpucast::math::bbox3d             bbox_t;

public : // methods

  void                    add             ( surface_ptr const& surface );
  void                    remove          ( surface_ptr const& surface );

  trim_approach_t         trim_approach   () const;
  void                    trim_approach   (trim_approach_t);

  std::size_t             surfaces        () const;
  std::size_t             trimcurves      () const;

  void                    clear           ();
  void                    merge           ( beziersurfaceobject const& );

  memory_usage            get_memory_usage() const;
  std::map<int, unsigned> order_surfaces  () const;
  std::map<int, unsigned> order_trimcurves() const;

  bbox_t                  bbox            () const;

  std::size_t             size            () const;
  const_surface_iterator  begin           () const;
  const_surface_iterator  end             () const;

  void                    print           ( std::ostream& os ) const;

  void                    init            ( unsigned subdivision_level_u = default_initial_subdivision,
                                            unsigned subdivision_level_v = default_initial_subdivision,
                                            unsigned fast_trim_texture_resolution = trim_preclassification_default_resolution);

  bool                    initialized     () const;

public : // getter for serialized data

  std::unordered_map<surface_ptr, unsigned> const&   serialized_vertex_base_indices() const;
  std::unordered_map<surface_ptr, unsigned> const&   serialized_obb_base_indices() const;
  std::unordered_map<surface_ptr, unsigned> const&   serialized_trim_base_indices() const;

  std::shared_ptr<trim_double_binary_serialization>  serialized_trimdata_as_double_binary() const;
  std::shared_ptr<trim_contour_binary_serialization> serialized_trimdata_as_contour_binary() const;
  std::shared_ptr<trim_kd_serialization>             serialized_trimdata_as_contour_kd() const;
  std::shared_ptr<trim_loop_list_serialization>      serialized_trimdata_as_contour_loop_list() const;

  std::vector<gpucast::math::vec4f> const&           serialized_controlpoints() const;

  std::vector<gpucast::math::vec3f> const&           serialized_raycasting_data_attrib0() const;
  std::vector<gpucast::math::vec4f> const&           serialized_raycasting_data_attrib1() const;
  std::vector<gpucast::math::vec4f> const&           serialized_raycasting_data_attrib2() const;
  std::vector<gpucast::math::vec4f> const&           serialized_raycasting_data_attrib3() const;
  std::vector<unsigned> const&                       serialized_raycasting_data_indices() const;
  
  std::vector<math::vec4f> const&                    serialized_tesselation_domain_buffer() const;
  std::vector<unsigned> const&                       serialized_tesselation_index_buffer() const;
  std::vector<patch_tesselation_data> const&         serialized_tesselation_attribute_data() const;
  std::vector<gpucast::math::vec4f> const&           serialized_tesselation_obbs() const;

private : // auxilliary methods

  void                    _clearbuffer    ();

  std::size_t             _serialize_control_points           ( gpucast::math::pointmesh2d<gpucast::math::point3d> const& points );

  std::size_t             _serialize_convex_hull               ( convex_hull const& chull );

  void                    _serialize_obb_data                  (surface_ptr const& surface);
  void                    _serialize_trimming_data             (trimdomain_ptr const& surface);
  void                    _serialize_raycasting_data           (surface_ptr const& surface);
  void                    _serialize_adaptive_tesselation_data (surface_ptr const& surface);

private : // data members

  /////////////////////////////////////////////////////////////////////////////
  // attributes
  /////////////////////////////////////////////////////////////////////////////
  surface_container                  _surfaces;
  bool                               _is_initialized = false;
  trim_approach_t                    _trim_approach = contour_kd_partition;

  unsigned                           _preclassification_resolution = trim_preclassification_default_resolution;
  unsigned                           _subdivision_u = default_initial_subdivision;
  unsigned                           _subdivision_v = default_initial_subdivision;
                                     
  /////////////////////////////////////////////////////////////////////////////
  // client side render information
  /////////////////////////////////////////////////////////////////////////////
  std::unordered_map<surface_ptr, unsigned>          _surface_vertex_base_ids;
  std::unordered_map<surface_ptr, unsigned>          _surface_obb_base_ids; 

  // data to change trimming method
  typedef std::map<trim_approach_t, unsigned> multi_trim_index_map;

  struct multi_trim_attrib_desc {
    std::size_t raycasting_base_index;
    std::size_t raycasting_vertex_count;
    std::size_t tesselation_base_index;
    unsigned trim_type;
  };

  std::map<trimdomain_ptr, multi_trim_index_map>     _domain_index_map;
  std::map<surface_ptr, multi_trim_attrib_desc>      _surface_index_map;
                                                     
  // storage for surface control points              
  std::vector<gpucast::math::vec4f>                  _control_points;

  // data for trimming
  std::shared_ptr<trim_double_binary_serialization>  _trimdata_double_binary_serialization;
  std::shared_ptr<trim_contour_binary_serialization> _trimdata_contour_binary_serialization;
  std::shared_ptr<trim_kd_serialization>             _trimdata_kd_serialization;
  std::shared_ptr<trim_loop_list_serialization>      _trimdata_loop_list_serialization;

  // data for arraybuffer
  struct ray_casting_data {
    std::vector<gpucast::math::vec3f>                  attribute_buffer_0;       // vertices of convex hulls
    std::vector<gpucast::math::vec4f>                  attribute_buffer_1;       // [start_u, start_v, 0, 0]
    std::vector<gpucast::math::vec4f>                  attribute_buffer_2;       // [trimid, dataid, orderu, orderv]
    std::vector<gpucast::math::vec4f>                  attribute_buffer_3;       // [umin, umax, vmin, vmax] bezierpatch-domain in nurbs-domainspace
                                                                     
    // data for element array buffer                                   
    std::vector<unsigned>                              index_buffer;       // indices of convex hulls
  };

  ray_casting_data                                    _ray_casting_data;

  // cpu ressources for adaptive tesselation
  struct tesselation_data {
    std::vector<math::vec4f>                         domain_buffer;         // domain corners
    std::vector<unsigned>                            index_buffer;          // index data
    std::vector<patch_tesselation_data>              patch_data_buffer;     // per patch attributes

    // "obbdata" -> object-oriented bbox [ center, low, high, 
    //                                     mat0, mat1, mat2, mat3, 
    //                                     invmat0, invmat1, invmat2, invmat3,
    //                                     LBF, RBF, RTF, LTF, 
    //                                     LBB, RBB, RTB, LTB ]
    std::vector<gpucast::math::vec4f>                obb_buffer;
  };

  tesselation_data                                    _tesselation_data;

};

} // namespace gpucast

#endif // GPUCAST_CORE_BEZIERSURFACEOBJECT_HPP
