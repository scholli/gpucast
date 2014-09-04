/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : matrix3x3.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_MATRIX3X3_HPP
#define GPUCAST_MATH_MATRIX3X3_HPP

// includes, system
#include <iosfwd> // fwd. decl: std::ostream

// includes, project

namespace gpucast { namespace math {

template<typename value_t>
class matrix3x3 
{
public:

  typedef value_t value_type;

public :

  matrix3x3                   ( );
  matrix3x3                   ( matrix3x3 const& );

  matrix3x3                   ( value_t a11, value_t a21, value_t a31,
                                value_t a12, value_t a22, value_t a32,
                                value_t a13, value_t a23, value_t a33
                              );

  explicit matrix3x3          ( value_t const [9]);

  ~matrix3x3                  ( );

  void swap                   ( matrix3x3& );

  matrix3x3&      operator=   ( matrix3x3 const& );

  matrix3x3&      operator*=  ( matrix3x3 const& );
  matrix3x3&      operator*=  ( value_t );
  matrix3x3&      operator/=  ( value_t );

  value_t&        operator[]  ( unsigned i );
  value_t const&  operator[]  ( unsigned i ) const;

  matrix3x3       adjoint     ( ) const;

  value_t         determinant ( ) const;

  bool            invert      ( );

  void            transpose   ( );

private:

  value_t data_[9];
};

template<typename value_t>
matrix3x3<value_t> operator-(matrix3x3<value_t> const&);

template<typename value_t>
matrix3x3<value_t> operator*(matrix3x3<value_t> const&, matrix3x3<value_t> const&);

template<typename value_t>
matrix3x3<value_t> operator*(matrix3x3<value_t> const&, value_t);

template<typename value_t>
matrix3x3<value_t> operator*(value_t, matrix3x3<value_t> const&);

template<typename value_t>
matrix3x3<value_t> operator/(matrix3x3<value_t> const&, value_t);

template<typename value_t, typename vec3_t>
vec3_t operator*(matrix3x3<value_t> const&, vec3_t const& );

template<typename value_t>
matrix3x3<value_t> inverse(matrix3x3<value_t> const&);

template<typename value_t>
matrix3x3<value_t> transpose(matrix3x3<value_t> const&);

template<typename value_t>
std::ostream& operator<<(std::ostream&, matrix3x3<value_t> const&);

// predefined typedefs
typedef matrix3x3<float>          matrix3f;
typedef matrix3x3<double>         matrix3d;

} } // namespace gpucast / namespace math

#include <gpucast/math/matrix3x3_impl.hpp>

#endif // GPUCAST_MATH_MATRIX3X3_HPP
