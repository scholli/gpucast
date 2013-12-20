/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : linear_curve_filter.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_LINEAR_CURVE_FILTER_HPP
#define GPUCAST_MATH_LINEAR_CURVE_FILTER_HPP

// includes, system

namespace gpucast { namespace math {

/////////////////////////////////////////////////////////////////////////////
// predicate class, e.g. for copy_if operations
/////////////////////////////////////////////////////////////////////////////
template <typename curve_ptr_type, unsigned DIMENSION>
class linear_curve_filter
{
  public : // typedef

    typedef typename curve_ptr_type::value_type   curve_type;
    typedef typename curve_type::value_type       value_type;

  public : // c'tor

    // predicate for copy_if-curve operations, optional epsilon 
    linear_curve_filter(value_type eps = 0)
      : _tolerance(eps)
    {}

    bool operator()(curve_ptr_type c)
    {
      return !c->is_constant(DIMENSION, _tolerance);
    }

    bool operator()(curve_type const& c)
    {
      return !c.is_constant(DIMENSION, _tolerance);
    }

  private : // attributes

    value_type    _tolerance;

};

} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_LINEAR_CURVE_FILTER_HPP
