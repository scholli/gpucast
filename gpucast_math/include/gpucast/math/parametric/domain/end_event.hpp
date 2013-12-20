/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : end_event.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_END_EVENT_HPP
#define GPUCAST_MATH_END_EVENT_HPP

#include <gpucast/math/parametric/domain/status_event.hpp>
#include <gpucast/math/parametric/domain/contour.hpp>
#include <gpucast/math/interval.hpp>

namespace gpucast { namespace math {

template <typename value_t>
class end_event : public status_event<value_t>
{
public : 

  typedef value_t              value_type;
  typedef interval<value_type> interval_type;
  using typename contour<value_type>::contour_segment_ptr;

public : 

  end_event ( value_type const& u, interval_type const& interval_v, contour_segment_ptr const& s )
    : _vertical_interval ( u, interval_v ),
      _segment           ( s )
  {}

  void update ( status_structure<point_t>& L, event_structure<point_t>& Q )
  {
    L.remove ( Q, location().x, line_ );
  }

  virtual void print ( std::ostream& os) const
  {
    os << "end event : ";
    status_event<point_t>::print(os);
  }

  virtual unsigned priority () const
  {
    return 2;
  }

  ~end_event()
  {}

private :

  contour_segment_ptr _segment;

};

} } // namespace gpucast / namespace math

#endif //GPUCAST_MATH_END_EVENT_HPP
