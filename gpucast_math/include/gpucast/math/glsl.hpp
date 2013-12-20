/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : glsl.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_GLSL_HPP
#define GPUCAST_MATH_GLSL_HPP

// header, system
#include <cmath>
#include <iostream>

#include <gpucast/math/matrix.hpp>
#include <gpucast/math/parametric/point.hpp>

namespace gpucast { namespace math {

  // typedefs
  typedef point<float, 4>     vec4;
  typedef point<float, 3>     vec3;
  typedef point<float, 2>     vec2;

  typedef point<unsigned, 2>  uvec2;
  typedef point<unsigned, 3>  uvec3;
  typedef point<unsigned, 4>  uvec4;
  
  typedef point<int, 4>       ivec4;
  typedef point<int, 3>       ivec3;
  typedef point<int, 2>       ivec2;

  typedef unsigned            uint;

  // template functions
  template <typename container_t>
  typename container_t::value_type imageLoad ( container_t const& c, int i )
  {
    return c.at(i);
  }

  // template functions
  template <typename container_t>
  void imageStore( container_t& c, int i, typename container_t::value_type const& v )
  {
    c.at(i) = v;
  }


} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_GLSL_HPP
