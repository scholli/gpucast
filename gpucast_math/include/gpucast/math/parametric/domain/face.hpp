/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : face.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_FACE_HPP
#define GPUCAST_MATH_FACE_HPP

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/domain/contour_segment.hpp>

namespace gpucast { namespace math {

template <typename value_t>
class face 
{
public :

  typedef value_t                                           value_type;
  typedef point<value_type, 2>                              point_type;
  typedef interval<value_type>                              interval_type;  
  typedef typename contour<value_type>::contour_segment_ptr contour_segment_ptr;

public :

  face ()
  {}

  ~face ()
  {}

  void horizontal_interval ( interval_type const& u )
  {
    _interval_u = u;
  }

  void vertical_interval ( interval_type const& u )
  {
    _interval_u = u;
  }

  void add ( contour_segment_ptr const& segment )
  {
    _segments.push_back ( segment );
  }

  std::vector<contour_segment_ptr> const& segments () const
  {
    return _segments;
  }

  void print(std::ostream& os ) const
  {
    os << " u: " << _interval_u 
       << ", v: " << _interval_v 
       << ", #segments : " << _segments.size() << std::endl;
  }

private :
    
  interval_type                   _interval_u;
  interval_type                   _interval_v;
  std::vector<contour_segment_ptr>  _segments;

};

template <typename value_t>
std::ostream& operator<< ( std::ostream& os, face<value_t> const& f )
{
  f.print(os);
  return os;
}

} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_FACE_HPP
