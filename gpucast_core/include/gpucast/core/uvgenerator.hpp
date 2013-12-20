/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : uvgenerator.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_UVGENERATOR_HPP
#define GPUCAST_CORE_UVGENERATOR_HPP

// header, system

// header, project
#include <gpucast/core/gpucast.hpp>


namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
template <typename vec4_type, unsigned first = 0, unsigned second = 1, unsigned third = 2, unsigned fourth = 3>
class uvgenerator
{
public : // typedefs

  typedef typename vec4_type::value_type value_type;

public : // c'tor

  uvgenerator ( std::size_t       order_u, 
                std::size_t       order_v,
	              value_type const& min_u,
                value_type const& max_u, 
                value_type const& min_v, 
                value_type const& max_v,
                value_type const& additional_information0 = 0, 
                value_type const& additional_information1 = 0
               )
    : _current_index( 0 ),
      _order_u    ( order_u ),
      _order_v    ( order_v ),
      _min_u      ( min_u),
      _min_v      ( min_v),
      _du         ( (max_u - min_u) / value_type(order_u - value_type(1) )),
      _dv         ( (max_v - min_v) / value_type(order_v - value_type(1) )),
      _info0      ( additional_information0 ),
      _info1      ( additional_information1 )
  {}

public : // operator

  vec4_type operator()()
  {
    vec4_type result;

    result[first]  = value_type(_min_u) + value_type(_current_index%_order_u) * _du;
		result[second] = value_type(_min_v) + value_type(_current_index/_order_u) * _dv;
		result[third]  = _info0;
    result[fourth] = _info1;

    ++_current_index;

    return result;
  }

private : // attributes

  // current index when enumerating rowwise
  std::size_t _current_index;

  // mesh size
  std::size_t _order_u;
  std::size_t _order_v;

  // according parameter values for patch
  value_type  _min_u;
  value_type  _min_v;

  // difference between adjacent control points
  value_type  _du;
  value_type  _dv;

  // add additional info into empty slots
  value_type  _info0;
  value_type  _info1;
};


}

#endif //
