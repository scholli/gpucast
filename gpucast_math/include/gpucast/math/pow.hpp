/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : pow.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_POW_HPP
#define GPUCAST_MATH_POW_HPP

// header, system

namespace gpucast { namespace math {

  template <int base, int exponent>
  struct meta_pow
  {
    enum { result = base * meta_pow<base, exponent-1>::result };
  };

  template <int base>
  struct meta_pow<base, 0>
  {
    enum { result = 1 };
  };

  template <int base>
  struct meta_pow<base, 1>
  {
    enum { result = base };
  };

} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_POW_HPP

