/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : matrix2x2.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_MATRIX2X2_HPP
#define GPUCAST_MATH_MATRIX2X2_HPP

// includes, system
#include <iosfwd> // fwd. decl: std::ostream

// includes, project

namespace gpucast { namespace math {

template<typename value_t>
class matrix2x2 
{
public:

  typedef value_t value_type;

public :

  matrix2x2                   ( );
  matrix2x2                   ( matrix2x2 const& );

  matrix2x2                   ( value_t a11, value_t a21,
                                value_t a12, value_t a22);

  explicit matrix2x2          ( value_t const [4]);

  ~matrix2x2                  ( );

  void swap                   ( matrix2x2& );

  matrix2x2&      operator=   ( matrix2x2 const& );

  matrix2x2&      operator*=  ( matrix2x2 const& );
  matrix2x2&      operator*=  ( value_t );
  matrix2x2&      operator/=  ( value_t );

  value_t&        operator[]  ( unsigned i );
  value_t const&  operator[]  ( unsigned i ) const;

  matrix2x2       adjoint     ( ) const;

  value_t         determinant ( ) const;

  bool            invert      ( );

  void            transpose   ( );

private:

  value_t data_[4];
};

template<typename value_t>
matrix2x2<value_t> operator-(matrix2x2<value_t> const&);

template<typename value_t>
matrix2x2<value_t> operator*(matrix2x2<value_t> const&, matrix2x2<value_t> const&);

template<typename value_t>
matrix2x2<value_t> operator*(matrix2x2<value_t> const&, value_t);

template<typename value_t>
matrix2x2<value_t> operator*(value_t, matrix2x2<value_t> const&);

template<typename value_t>
matrix2x2<value_t> operator/(matrix2x2<value_t> const&, value_t);

template<typename value_t, typename vec2_t>
vec2_t operator*(matrix2x2<value_t> const&, vec2_t const& );

template<typename value_t>
matrix2x2<value_t> inverse(matrix2x2<value_t> const&);

template<typename value_t>
matrix2x2<value_t> transpose(matrix2x2<value_t> const&);

template<typename value_t>
std::ostream& operator<<(std::ostream&, matrix2x2<value_t> const&);

// predefined typedefs
typedef matrix2x2<float>          matrix2f;
typedef matrix2x2<double>         matrix2d;

} } // namespace gpucast / namespace math

#include <gpucast/math/matrix2x2_impl.hpp>

#endif // GPUCAST_MATH_MATRIX3X3_HPP
