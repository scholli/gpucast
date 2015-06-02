/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : signed_gradient.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_SIGNED_GRADIENT_HPP
#define GPUCAST_MATH_SIGNED_GRADIENT_HPP

#include <gpucast/math/util/polar_conversion.hpp>

namespace gpucast { namespace math {
  namespace util {

  /////////////////////////////////////////////////////////////////////////////
  // given a signed gradient and asssuming that a positive gradient means 'inside'
  // this method classifies a given sample point if it also lies 'inside'
  /////////////////////////////////////////////////////////////////////////////
  template <typename vec2_t>
  class classify_sample_by_signed_gradient
  {
  public : 
    typedef typename vec2_t::value_type value_type;

    bool operator()(vec2_t const& angle_radius, 
                    vec2_t const& sample) const
    {
      polar_to_euclid<vec2_t> r2e;

      vec2_t gradient = r2e(angle_radius);
      vec2_t normalized_gradient = gradient / gradient.abs();

      value_type distance = dot(normalized_gradient, sample - gradient);

      return (distance < 0) ^ (angle_radius[1] < 0);
    }
  };


  } // namespace util
} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_SIGNED_GRADIENT_HPP
