/********************************************************************************
*
* Copyright (C) 2008 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziersurface.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMMED_BEZIERSURFACE_HPP
#define GPUCAST_CORE_TRIMMED_BEZIERSURFACE_HPP

// header, system
#include <vector> // std::vector

// header, external
#include <gpucast/math/parametric/beziersurface.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/oriented_boundingbox.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/convex_hull.hpp>
#include <gpucast/core/trimdomain.hpp>



namespace gpucast {

class beziersurfaceobject;

class GPUCAST_CORE beziersurface : public gpucast::math::beziersurface3d
{
public : // enums, typedefs

  typedef gpucast::math::beziersurface3d         base_type;
  typedef gpucast::math::pointmesh2d<point_type> mesh_type;

  typedef trimdomain::point_type                 curve_point_type;
  typedef trimdomain::curve_type                 curve_type;
  typedef std::vector<trimdomain::curve_type>    curve_container;
  
  typedef std::shared_ptr<beziersurfaceobject>   host_ptr;
  typedef std::shared_ptr<trimdomain>            trimdomain_ptr;

public : // c'tors and d'tor

  beziersurface     ( );

  beziersurface     ( gpucast::math::beziersurface3d const& untrimmed_surface );

  void                          print          ( std::ostream& os ) const;

public : // methods

  /// add a trimming loop
  void                          add            ( curve_container const& trimloop );

  /* virtual */ void            split          ( beziersurface& bl,
                                                 beziersurface& tl,
                                                 beziersurface& br,
                                                 beziersurface& tr ) const;

  void                          trimtype       ( bool type );

  /// set the parameter range of the nurbs surface the bezier surface belongs to (there might be trimming curves that effect this surface)
  void                          nurbsdomain    ( trimdomain::bbox_type const& );

  /// set parameter range of this patch (u,v in [0.0,1.0])
  void                          bezierdomain   ( trimdomain::bbox_type const& );
  trimdomain::bbox_type const&  bezierdomain () const;

  void                          preprocess     ( std::size_t subdivision_level_u, 
                                                 std::size_t subdivision_level_v );

  void                          tesselate      ( std::vector<gpucast::math::vec3f>& vertices,
                                                 std::vector<int>&         indices ) const;

  void                          update         ();

  bool                          initialized    () const;

  bool                          trimtype       ( ) const;
  std::size_t                   trimcurves     ( ) const;

  math::obbox3d const&          obb            ( ) const;

  trimdomain_ptr const&         domain         ( ) const;
  void                          domain         ( trimdomain_ptr const& domain );

  convex_hull const&            convexhull     ( ) const;
  mesh_type const&              points         ( ) const;

private : // attributes

  convex_hull             _chull;
  trimdomain_ptr          _trimdomain;
  trimdomain::bbox_type   _bezierdomain; // umin, umax, vmin, vmax
  math::obbox3d           _obb;
};

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMMED_BEZIERSURFACE_HPP
