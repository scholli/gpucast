/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : status_event.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_STATUS_EVENT_HPP
#define GPUCAST_MATH_STATUS_EVENT_HPP

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/interval.hpp>

namespace gpucast { namespace math {

template <typename T> class status_structure;
template <typename T> class event_structure;

template <typename value_t>
class status_event
{
public :

  typedef value_t              value_type;
  typedef interval<value_type> interval_type;

public :

  status_event ( value_t const& u, interval_type const& interval_v )
    : _position_u ( u ),
      _interval_v ( interval_v )
  {}

  virtual ~status_event()
  {}

  virtual void update ( status_structure<value_type>& L, event_structure<value_type>& Q ) = 0;

  virtual void print ( std::ostream& os) const
  {
    os << "u: " << _position_u << " , v-interval : " << _interval_v << std::endl;
  }

  virtual unsigned priority () const = 0;

  value_type const& horizontal_position () const
  {
    return horizontal_position;
  }

  interval_type const& vertical_interval () const
  {
    return _vertical_interval;
  }

private :

  value_type    _position_u;
  interval_type _interval_v;

};

} } // namespace gpucast / namespace math

#endif //GPUCAST_MATH_STATUS_EVENT_HPP
