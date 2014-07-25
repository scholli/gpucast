/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : uvwgenerator.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_UVWGENERATOR_HPP
#define GPUCAST_UVWGENERATOR_HPP

// header, system

// header, project
#include <gpucast/volume/gpucast.hpp>


namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
template <typename vec4_type, unsigned first = 0, unsigned second = 1, unsigned third = 2, unsigned fourth = 3>
class uvwgenerator
{
public : // typedefs

  typedef typename vec4_type::value_type value_type;

public : // c'tor

  uvwgenerator ( std::size_t       order_u, 
                 std::size_t       order_v,
                 std::size_t       order_w,
	               value_type const& min_u,
                 value_type const& max_u, 
                 value_type const& min_v, 
                 value_type const& max_v,
                 value_type const& min_w, 
                 value_type const& max_w,
                 value_type const& additional_information0 = 0
                )
    : _current_index( 0 ),
      _order_u    ( order_u ),
      _order_v    ( order_v ),
      _order_w    ( order_w ),
      _min_u      ( min_u),
      _min_v      ( min_v),
      _min_w      ( min_w),
      _du         ( (max_u - min_u) / value_type(order_u - value_type(1) )),
      _dv         ( (max_v - min_v) / value_type(order_v - value_type(1) )),
      _dw         ( (max_v - min_w) / value_type(order_w - value_type(1) )),
      _info0      ( additional_information0 )
  {}

public : // operator

  vec4_type operator()()
  {
    vec4_type result;

    result[first]  = value_type(_min_u) + value_type(_current_index%_order_u) * _du;
		result[second] = value_type(_min_v) + value_type((_current_index%(_order_u*_order_v))/_order_u) * _dv;
		result[third]  = value_type(_min_w) + value_type(_current_index/(_order_u*_order_v)) * _dw;
    result[fourth] = _info0;

    ++_current_index;

    return result;
  }

private : // attributes

  // current index when enumerating rowwise
  std::size_t _current_index;

  // mesh size
  std::size_t _order_u;
  std::size_t _order_v;
  std::size_t _order_w;

  // according parameter values for patch
  value_type  _min_u;
  value_type  _min_v;
  value_type  _min_w;

  // difference between adjacent control points
  value_type  _du;
  value_type  _dv;
  value_type  _dw;

  // add additional info into empty slots
  value_type  _info0;
};



}

#endif //
