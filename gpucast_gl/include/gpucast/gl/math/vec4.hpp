/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vec4.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_VEC4_HPP
#define GPUCAST_GL_VEC4_HPP

// header, system
#include <iostream> // std::cout
#include <cassert> // assert

// header, project
#include <gpucast/gl/math/vec2.hpp>
#include <gpucast/gl/math/vec3.hpp>

namespace gpucast { namespace gl {

template <typename value_t>
class vec4
{

public : // enums and typedefs

  typedef value_t value_type;
  static size_t const size = 4;

public :

  vec4                    ( );

  explicit vec4           ( value_t x,
                            value_t y,
                            value_t z,
                            value_t w );

  template <typename indexed_array_t>
  vec4                    ( indexed_array_t const& );

  vec4                    ( vec4 const& rhs );

  template <typename iterator_t>
  vec4                    ( iterator_t beg,
                            iterator_t end );

  ~vec4                   ( );


public : // operators

  value_t const& operator[]   ( unsigned axis) const;
  value_t&       operator[]   ( unsigned axis);

  vec4&       operator=       ( vec4 const& rhs);

  template <typename indexed_array_t>
  vec4&        operator=      ( indexed_array_t const& );

  void        operator*=      ( value_t     scalar);
  void        operator/=      ( value_t     scalar);
  void        operator-=      ( vec4 const& rhs);
  void        operator+=      ( vec4 const& rhs);


public : // methods

  void        swap            ( vec4&          rhs);
  void        print           ( std::ostream&  os) const;

  vec4        as_homogenous   ( ) const; // P[wx, wy, w]
  vec4        as_euclidian    ( ) const; // P[x/w, y/w, w]

  value_t     abs             ( ) const; // length
  value_t     length          ( ) const;
  value_t     length_square   ( ) const;

  void        normalize       ( );

  static vec4 minimum         ( );
  static vec4 maximum         ( );

  value_t       x             ( ) const;
  value_t       y             ( ) const;
  value_t       z             ( ) const;
  value_t       w             ( ) const;

  vec2<value_t> xy            ( ) const;

  vec3<value_t> xyz           ( ) const;

private : // data members

  value_t _data[size];
};

template <typename value_t>
bool operator==               ( vec4<value_t> const& lhs, vec4<value_t> const& rhs );

template <typename value_t>
bool operator!=               ( vec4<value_t> const& lhs, vec4<value_t> const& rhs );

template <typename value_t>
vec4<value_t> elementwise_min ( vec4<value_t> const&, vec4<value_t> const& );

template <typename value_t>
vec4<value_t> elementwise_max ( vec4<value_t> const&, vec4<value_t> const& );

template <typename value_t>
std::ostream& operator<<(std::ostream& os, vec4<value_t> const& rhs);

template <typename value_t>
vec4<value_t> operator*(vec4<value_t> const& lhs, value_t const& rhs);

template <typename value_t>
vec4<value_t> operator*(value_t const& lhs, vec4<value_t> const& rhs);

template <typename value_t>
vec4<value_t> operator/(vec4<value_t> const& lhs, value_t const& rhs);

template <typename value_t>
vec4<value_t> operator+(vec4<value_t> const& lhs, vec4<value_t> const& rhs);

template <typename value_t>
value_t  dot(vec4<value_t> const& lhs, vec4<value_t> const& rhs);

// typedefs
typedef vec4<float>     vec4f;
typedef vec4<double>    vec4d;
typedef vec4<int>       vec4i;
typedef vec4<unsigned>  vec4u;
typedef vec4<char>      vec4b;

} } // namespace gpucast / namespace gl

#include <gpucast/gl/math/vec4_impl.hpp>

#endif // GPUCAST_GL_vec4_HPP
