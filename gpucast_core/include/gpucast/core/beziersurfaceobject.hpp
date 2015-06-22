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
#ifndef GPUCAST_CORE_BEZIEROBJECT_HPP
#define GPUCAST_CORE_BEZIEROBJECT_HPP

// header, system

// header, external
#include <gpucast/math/axis_aligned_boundingbox.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/beziersurface.hpp>
#include <gpucast/core/surface_renderer.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

class surface_renderer;

class GPUCAST_CORE beziersurfaceobject
{
public : // friends

  friend class surface_renderer_gl;

public : // enums, typedefs

  enum trim_approach_t {
    no_trimming     = 0,
    double_binary   = 1,
    contours_binary = 2,
    contours_kd     = 3,
    loop_list       = 4,
    count           = 5
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

  void                    udpate          ();

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

  void                    init            ( std::size_t subdivision_level_u = 0,
                                            std::size_t subdivision_level_v = 0 );

  bool                    initialized     () const;

private : // auxilliary methods

  void                    _clearbuffer     ();

  std::size_t             _add            ( surface_ptr const&                                                                          surface,
                                            std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type>&                  db_domains,
                                            std::unordered_map<curve_ptr, trimdomain_serializer::address_type>&                       db_curves,
                                            std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type>&                  cmb_domains,
                                            std::unordered_map<trimdomain::contour_segment_ptr, trimdomain_serializer::address_type>& cmb_segments, 
                                            std::unordered_map<curve_ptr, trimdomain_serializer::address_type>&                       cmb_curves);

  std::size_t             _add            ( gpucast::math::pointmesh2d<gpucast::math::point3d> const& points );

  std::size_t             _add            ( convex_hull const& chull );

public : // data members

  /////////////////////////////////////////////////////////////////////////////
  // attributes
  /////////////////////////////////////////////////////////////////////////////
  surface_container                  _surfaces;
  bool                               _is_initialized = false;
                                     
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
  std::vector<unsigned>                     _indices; // indices of convex hulls

  // data for texturebuffer
  std::vector<gpucast::math::vec4f>    _controlpoints; // "vertexdata" -> control point data for texturebuffer

  // trim approach 1 : contour based trimming
  std::vector<gpucast::math::vec4f>    _cmb_partition;    
  std::vector<gpucast::math::vec4f>    _cmb_contourlist;     
  std::vector<gpucast::math::vec4f>    _cmb_curvelist;    
  std::vector<float>                   _cmb_curvedata;    
  std::vector<gpucast::math::vec3f>    _cmb_pointdata;    

  // trim approach 2 : double binary search map
  std::vector<gpucast::math::vec4f>    _db_partition;   // "trimdata"    
  std::vector<gpucast::math::vec4f>    _db_celldata;    // "urangeslist" 
  std::vector<gpucast::math::vec4f>    _db_curvelist;   // "curvelist"   
  std::vector<gpucast::math::vec3f>    _db_curvedata;   // "curvedata"   
};

} // namespace gpucast

#endif // GPUCAST_CORE_BEZIEROBJECT_HPP
