/********************************************************************************
*
* Copyright (C) 2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : gcd.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_GCD_HPP
#define GPUCAST_MATH_GCD_HPP

// header, system

namespace gpucast { namespace math {

  template <long A, long B>
  struct gcd_t
  {
    enum { result = gcd_t<B, A%B>::result };
  };

  template <long A>
  struct gcd_t<A, 1>
  {
    enum { result = 1 };
  };

  template <long A>
  struct gcd_t<A, 0>
  {
    enum { result = A + 2*A*(-long(A<0)) };
  };

  long gcd ( long A, long B )
  {
    return (A % B == 0) ? abs(B) : gcd(B, A%B);
  }

} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_GCD_HPP

