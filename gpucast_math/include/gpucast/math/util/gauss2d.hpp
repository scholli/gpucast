/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : gauss2d.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_GAUSS2D_HPP
#define GPUCAST_MATH_GAUSS2D_HPP

#ifndef M_PI 
#define M_PI 3.141592653589793238462
#endif

namespace gpucast { namespace math {
  namespace util {

  /////////////////////////////////////////////////////////////////////////////
  template <typename vec2_t>
  class gauss2d
  {
  public :
    typedef typename vec2_t::value_type value_type;

    gauss2d (value_type sigma = 0.5)
      : _sigma(sigma)
    {}

    value_type operator()(vec2_t v) const
    {
      value_type x = v[0];
      value_type y = v[1];
      return (value_type(1) / (sqrt(value_type(2) * value_type(M_PI)) * _sigma)) * exp(-((x*x + y*y)/(value_type(2)*_sigma*_sigma)));
    }

  private : 
    value_type _sigma;
  };


  } // namespace util
} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_GAUSS2D_HPP
