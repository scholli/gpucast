/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbssurface.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMMED_NURBSSURFACE_HPP
#define GPUCAST_CORE_TRIMMED_NURBSSURFACE_HPP

// header, system
#include <vector>

// header external
#include <gpucast/math/parametric/nurbscurve.hpp>
#include <gpucast/math/parametric/nurbssurface.hpp>
#include <gpucast/math/parametric/point.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>


namespace gpucast {

class GPUCAST_CORE nurbssurface : public gpucast::math::nurbssurface<gpucast::math::point3d>
{
public : // typedefs, enums

  typedef double                                      value_type;
  typedef gpucast::math::point<value_type,3>                    point_type;
  typedef gpucast::math::nurbssurface<point_type>               surface_type;
  typedef gpucast::math::nurbscurve<gpucast::math::point<value_type,2> >  curve_type;

  typedef std::vector<curve_type>                     curve_container;
  typedef curve_container::iterator                   curve_iterator;
  typedef curve_container::const_iterator             const_curve_iterator;

  typedef std::vector<curve_container>                trimloop_container;
  typedef trimloop_container::iterator                trimloop_iterator;
  typedef trimloop_container::const_iterator          const_trimloop_iterator;

public : // c'tor / d'tor

  nurbssurface  ();
  nurbssurface  ( nurbssurface const& rhs );
  ~nurbssurface ();

  void                      swap        ( nurbssurface& rhs );
  nurbssurface&             operator=   ( nurbssurface const& rhs );

public : // methods

  void                      add         ( curve_container const& tc );

  trimloop_container const& trimloops   () const;
  
  void                      print       ( std::ostream& os) const;

  void                      trimtype    ( bool type );
  bool                      trimtype    () const;

private : // data members

  trimloop_container      _trimloops;
  bool                    _trimtype; // 0 = trim inner, 1 = trim outer
};


  /*template <typename iterator_t>
  inline void
  nurbssurface::trimcurves(iterator_t begin, iterator_t end)
  {
    _trimcurves.resize(std::distance(begin, end));
    std::copy(begin, end, _trimcurves.begin());
  }*/

  std::ostream& operator<<(std::ostream& os, nurbssurface const& rhs);

} // namespace gpucast

#endif // GPUCAST_CORE_NURBSSURFACE_HPP
