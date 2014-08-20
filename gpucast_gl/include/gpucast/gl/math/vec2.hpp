/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vec2.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_VEC2_HPP
#define GPUCAST_GL_VEC2_HPP

// header, system
#include <iostream> // std::cout
#include <cassert>  // assert
#include <limits>

// header, project


namespace gpucast { namespace gl {

template <typename value_t>
class vec2
{

public : // enums and typedefs

  typedef value_t value_type;
  static size_t const size = 2;

public :

  vec2                    ( );

  explicit vec2           ( value_t x,
                            value_t y );

  template <typename indexed_array_t>
  vec2                    ( indexed_array_t const& );

  vec2                    ( vec2 const& rhs );

  ~vec2                   ( );


public : // operators

  value_t const& operator[]   ( unsigned axis) const;
  value_t&       operator[]   ( unsigned axis);

  vec2&       operator=       ( vec2 const& rhs);

  template <typename indexed_array_t>
  vec2&        operator=      ( indexed_array_t const& );


  vec2&       operator*=      ( value_t     scalar);
  vec2&       operator/=      ( value_t     scalar);
  vec2&       operator-=      ( vec2 const& rhs);
  vec2&       operator+=      ( vec2 const& rhs);



public : // methods

  void        swap            ( vec2&          rhs);
  void        print           ( std::ostream&  os) const;
 
  vec2        as_homogenous   ( ) const; // P[wx, wy, w]
  vec2        as_euclidian    ( ) const; // P[x/w, y/w, w]

  value_t     abs             ( ) const; // length
  value_t     length          ( ) const;
  value_t     length_square   ( ) const;

  void        normalize       ( );

  static vec2 minimum         ( );
  static vec2 maximum         ( );

private : // data members

  value_t _data[size];
};

template <typename value_t>
bool operator==        ( vec2<value_t> const& lhs, vec2<value_t> const& rhs );

template <typename value_t>
bool operator!=        ( vec2<value_t> const& lhs, vec2<value_t> const& rhs );

template <typename value_t>
vec2<value_t> operator/( vec2<value_t> const&, vec2<value_t> const& );

template <typename value_t>
vec2<value_t> operator*( vec2<value_t> const&, vec2<value_t> const& );

template <typename value_t>
vec2<value_t> operator+( vec2<value_t> const&, vec2<value_t> const& );

template <typename value_t>
vec2<value_t> operator-( vec2<value_t> const&, vec2<value_t> const& );

template <typename value_t>
vec2<value_t> elementwise_min ( vec2<value_t> const&, vec2<value_t> const& );

template <typename value_t>
vec2<value_t> elementwise_max ( vec2<value_t> const&, vec2<value_t> const& );

template <typename value_t>
value_t length ( vec2<value_t> const& );

template <typename value_t>
std::ostream& operator<<(std::ostream& os, vec2<value_t> const& rhs);

template <typename value_t>
vec2<value_t> operator*(vec2<value_t> const& lhs, value_t const& rhs);

template <typename value_t>
vec2<value_t> operator*(value_t const& lhs, vec2<value_t> const& rhs);

template <typename value_t>
vec2<value_t> operator/(vec2<value_t> const& lhs, value_t const& rhs);

template <typename value_t>
value_t  dot(vec2<value_t> const& lhs, vec2<value_t> const& rhs);

// typedefs
typedef vec2<float>     vec2f;
typedef vec2<double>    vec2d;
typedef vec2<int>       vec2i;
typedef vec2<char>      vec2b;

} } // namespace gpucast / namespace gl

#include <gpucast/gl/math/vec2_impl.hpp>

#endif // GPUCAST_GL_vec2_HPP
