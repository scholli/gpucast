/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vec3.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_VEC3_HPP
#define GPUCAST_GL_VEC3_HPP

// header, system
#include <iostream> // std::cout
#include <cassert> // assert

// header, project


namespace gpucast { namespace gl {

template <typename value_t>
class vec3
{
public : // enums and typedefs

  typedef value_t value_type;
  static size_t const size = 3;

public :

  vec3                    ( );

  explicit vec3           ( value_t x,
                            value_t y,
                            value_t z );

  template <typename indexed_array_t>
  vec3                    ( indexed_array_t const& );

  vec3                    ( vec3 const& rhs );

  template <typename iterator_t>
  vec3                    ( iterator_t beg,
                            iterator_t end );

  ~vec3                   ( );


public : // operators

  value_t const& operator[]   ( unsigned axis) const;
  value_t&       operator[]   ( unsigned axis);

  vec3&       operator=       ( vec3 const& rhs);

  template <typename indexed_array_t>
  vec3&        operator=      ( indexed_array_t const& );

  vec3&       operator*=      ( value_t     scalar);
  vec3&       operator/=      ( value_t     scalar);
  vec3&       operator-=      ( vec3 const& rhs);
  vec3&       operator+=      ( vec3 const& rhs);

public : // methods

  void        swap            ( vec3&          rhs);
  void        print           ( std::ostream&  os) const;

  vec3        as_homogenous   ( ) const; // P[wx, wy, w]
  vec3        as_euclidian    ( ) const; // P[x/w, y/w, w]

  value_t     abs             ( ) const; // length
  value_t     length          ( ) const;
  value_t     length_square   ( ) const;

  void        normalize       ( );

  static vec3 minimum         ( );
  static vec3 maximum         ( );

private : // data members

  value_t     _data[size];
};

template <typename value_t>
bool operator==               ( vec3<value_t> const& lhs, vec3<value_t> const& rhs );

template <typename value_t>
bool operator!=               ( vec3<value_t> const& lhs, vec3<value_t> const& rhs );

template <typename value_t>
vec3<value_t> elementwise_min ( vec3<value_t> const&, vec3<value_t> const& );

template <typename value_t>
vec3<value_t> elementwise_max ( vec3<value_t> const&, vec3<value_t> const& );

template <typename value_t>
vec3<value_t> cross           ( vec3<value_t> const& lhs, vec3<value_t> const& rhs );

template <typename value_t>
std::ostream& operator<<(std::ostream& os, vec3<value_t> const& rhs);

template <typename value_t>
vec3<value_t> operator*(vec3<value_t> const& lhs, value_t const& rhs);

template <typename value_t>
vec3<value_t> operator*(value_t const& lhs, vec3<value_t> const& rhs);

template <typename value_t>
vec3<value_t> operator/(vec3<value_t> const& lhs, value_t const& rhs);

template <typename value_t>
vec3<value_t>operator+(vec3<value_t> const& lhs, vec3<value_t> const& rhs);

template <typename value_t>
value_t dot(vec3<value_t> const& lhs, vec3<value_t> const& rhs);

// typedefs
typedef vec3<float>     vec3f;
typedef vec3<double>    vec3d;
typedef vec3<int>       vec3i;
typedef vec3<unsigned>  vec3u;
typedef vec3<char>      vec3b;

} } // namespace gpucast / namespace gl

#include <gpucast/gl/math/vec3_impl.hpp>

#endif // GPUCAST_GL_VEC3_HPP
