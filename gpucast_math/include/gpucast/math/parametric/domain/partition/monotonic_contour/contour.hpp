/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_CONTOUR_HPP
#define GPUCAST_MATH_CONTOUR_HPP

// includes, system
#include <cassert>

// includes. projects
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_segment.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
class contour 
{
public : // enums / typedefs

  typedef value_t                                  value_type;
  typedef point<value_type,2>                      point_type;
  typedef axis_aligned_boundingbox<point_type>     bbox_type;

  typedef beziercurve<point_type>                  curve_type;
  typedef std::shared_ptr<curve_type>              curve_ptr;

  typedef contour_segment<value_type>              contour_segment_type;
  typedef std::shared_ptr<contour_segment_type>    contour_segment_ptr;

  typedef std::vector<curve_ptr>                   curve_container;
  typedef typename curve_container::iterator       curve_iterator;
  typedef typename curve_container::const_iterator const_curve_iterator;

public : // c'tor / d'tor

  template <typename curve_ptr_iterator_t>
  contour  ( curve_ptr_iterator_t begin, curve_ptr_iterator_t end );

  ~contour ();

public : // methods

  // check if contour defines a non-overlapping. piecewise continous and closed boundary
  bool                  valid () const;
                              
  bool                  empty () const;
  std::size_t           size  () const;

  const_curve_iterator  begin () const;
  const_curve_iterator  end   () const;

  curve_container const& curves() const;
  bbox_type             bbox() const;

  bool                  is_inside(point_type const& origin) const;
  bool                  is_inside(contour const& other) const;

  // split contour into bi-monotonic pieces
  void                  monotonize ();
  std::vector<contour_segment_ptr> const& monotonic_segments() const;

  // print to output stream
  void print ( std::ostream& os ) const;

private :

  std::vector<contour_segment_ptr> _monotonic_segments;
  curve_container                  _curves;
};

template <typename value_t>
std::ostream& operator<<(std::ostream& os,  contour<value_t> const& );

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#include "contour_impl.hpp"

#endif // GPUCAST_MATH_CONTOUR_HPP
