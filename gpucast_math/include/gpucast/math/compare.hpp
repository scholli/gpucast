/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : compare.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_COMPARE_HPP
#define GPUCAST_MATH_COMPARE_HPP

// header, system

namespace gpucast { namespace math {

  template <typename value_type>
  bool weak_equal(value_type const& lhs, value_type const rhs, value_type const& epsilon)
  {
    return fabs(rhs - lhs) < epsilon;
  }

} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_COMPARE_HPP

