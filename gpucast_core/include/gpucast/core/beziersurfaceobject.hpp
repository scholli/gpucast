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

public : // enums, typedefs

  enum trim_approach_t {
    no_trimming              = 0,
    curve_binary_partition   = 1,
    contour_binary_partition = 2,
    contour_kd_partition     = 3,
    contour_list             = 4,
    count                    = 5
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

  std::map<int, unsigned> order_surfaces  () const;
  std::map<int, unsigned> order_trimcurves() const;

  bbox_t                  bbox            () const;

  std::size_t             size            () const;
  const_surface_iterator  begin           () const;
  const_surface_iterator  end             () const;

  void                    print           ( std::ostream& os ) const;

  void                    init            ( unsigned subdivision_level_u = 0,
                                            unsigned subdivision_level_v = 0,
                                            unsigned fast_trim_texture_resolution = 0 );

  bool                    initialized     () const;

public : // getter for serialized data

  std::shared_ptr<trim_double_binary_serialization>  serialized_trimdata_as_double_binary() const;
  std::shared_ptr<trim_contour_binary_serialization> serialized_trimdata_as_contour_binary() const;
  std::shared_ptr<trim_kd_serialization>             serialized_trimdata_as_contour_kd() const;
  std::shared_ptr<trim_loop_list_serialization>      serialized_trimdata_as_contour_loop_list() const;

  std::vector<gpucast::math::vec3f> const& serialized_raycasting_data_attrib0() const;
  std::vector<gpucast::math::vec4f> const& serialized_raycasting_data_attrib1() const;
  std::vector<gpucast::math::vec4f> const& serialized_raycasting_data_attrib2() const;
  std::vector<gpucast::math::vec4f> const& serialized_raycasting_data_attrib3() const;
  std::vector<gpucast::math::vec4f> const& serialized_raycasting_data_controlpoints() const;
  std::vector<gpucast::math::vec4f> const& serialized_raycasting_data_obbs() const;
  std::vector<unsigned> const& serialized_raycasting_data_indices() const;
  
private : // auxilliary methods

  void                    _clearbuffer    ();

  std::size_t             _add            ( surface_ptr surface, unsigned fast_trim_texture_resolution );

  std::size_t             _add            ( gpucast::math::pointmesh2d<gpucast::math::point3d> const& points );

  std::size_t             _add            ( convex_hull const& chull );

private : // data members

  /////////////////////////////////////////////////////////////////////////////
  // attributes
  /////////////////////////////////////////////////////////////////////////////
  surface_container                  _surfaces;
  bool                               _is_initialized = false;
  trim_approach_t                    _trim_approach = contour_kd_partition;
                                     
  std::size_t                        _subdivision_u;
  std::size_t                        _subdivision_v;
                                     
  /////////////////////////////////////////////////////////////////////////////
  // client side render information
  /////////////////////////////////////////////////////////////////////////////

  // data for arraybuffer
  std::vector<gpucast::math::vec3f>    _attrib0; // vertices of convex hulls
  std::vector<gpucast::math::vec4f>    _attrib1; // [start_u, start_v, 0, 0]
  std::vector<gpucast::math::vec4f>    _attrib2; // [trimid, dataid, orderu, orderv]
  std::vector<gpucast::math::vec4f>    _attrib3; // [umin, umax, vmin, vmax] bezierpatch-domain in nurbs-domainspace

  // data for element array buffer
  std::vector<unsigned>                _indices; // indices of convex hulls

  // data for texturebuffer
  std::vector<gpucast::math::vec4f>    _controlpoints; // "vertexdata" -> control point data for texturebuffer
  std::vector<gpucast::math::vec4f>    _obbs;          // "obbdata" -> object-oriented bbox [ center, low, high, 
                                                       //                                     mat0, mat1, mat2, mat3, 
                                                       //                                     invmat0, invmat1, invmat2, invmat3, 
                                                       //                                     LBF, RBF, RTF, LTF, 
                                                       //                                     LBB, RBB, RTB, LTB ]

  // data for trimming
  std::shared_ptr<trim_double_binary_serialization>  _trimdata_double_binary_serialization;
  std::shared_ptr<trim_contour_binary_serialization> _trimdata_contour_binary_serialization;
  std::shared_ptr<trim_kd_serialization>             _trimdata_kd_serialization;
  std::shared_ptr<trim_loop_list_serialization>      _trimdata_loop_list_serialization;

  // data to change trimming method
  struct multi_trim_index {
    std::size_t double_binary_index;
    std::size_t contour_binary_index;
    std::size_t contour_kd_index;
    std::size_t loop_list_index;
  };

  struct multi_trim_attrib_desc {
    std::size_t base_index;
    std::size_t count;
    unsigned trim_type;
  };

  std::map<trimdomain_ptr, multi_trim_index>       _domain_index_map;
  std::map<surface_ptr, multi_trim_attrib_desc>    _surface_index_map;
};

} // namespace gpucast

#endif // GPUCAST_CORE_BEZIERSURFACEOBJECT_HPP
