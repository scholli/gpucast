/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : maximal_coordinates.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_MAXIMAL_COORDINATES_HPP
#define GPUCAST_MATH_MAXIMAL_COORDINATES_HPP

#include <limits>


namespace gpucast { namespace math {

  template<typename point_t>
  class maximal_coordinates {
  public :

    typedef typename point_t::value_type value_type;

  public :
    maximal_coordinates(std::size_t coord)
      : _max   ( -std::numeric_limits<value_type>::max() ),
        _coord ( coord )
    {}

    void operator()(point_t const& p)
    {
      _max = std::max(_max, p[_coord]);
    }

    value_type const& result() const
    {
      return _max;
    }

  private :
    value_type  _max;
    std::size_t _coord;
  };

} } // namespace gpucast / namespace math

#endif
