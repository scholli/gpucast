   /********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbssurfaceobject.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_NURBSOBJECT_HPP
#define GPUCAST_CORE_NURBSOBJECT_HPP

#ifdef _MSC_VER
 #pragma warning(disable: 4251)
#endif

// header, system
#include <vector>

// header, external
#include <boost/optional.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/nurbssurface.hpp>
#include <gpucast/math/vec3.hpp>


namespace gpucast {

// simple container for all nurbssurfaces of Object
class GPUCAST_CORE nurbssurfaceobject
{
public : // enums, typedefs

  typedef nurbssurface                          surface_type;
  typedef std::vector<nurbssurface>             surface_container;
  typedef surface_type::curve_type              curve_type;

  typedef surface_type::curve_iterator          curve_iterator;
  typedef surface_type::const_curve_iterator    const_curve_iterator;

  typedef surface_container::iterator           iterator;
  typedef surface_container::const_iterator     const_iterator;

public : // c'tor / d'tor

  nurbssurfaceobject();
  ~nurbssurfaceobject();

public : // methods

  // add a rational surface to scene
  void                add     ( nurbssurface const& nrbs );

  // print
  void                print   ( std::ostream& os ) const;
  
  const_iterator      begin   () const;
  const_iterator      end     () const;

  iterator            begin   ();
  iterator            end     ();

  std::size_t         surfaces   () const;
  std::size_t         trimcurves () const;

  boost::optional<math::vec3f> color() const { return _color; };
  void                color(math::vec3f const& c) { _color = c; };

private : // data members

  surface_container   _surfaces;
  boost::optional<math::vec3f> _color;
};

} // namespace gpucast

#endif // GPUCAST_CORE_NURBSOBJECT_HPP
