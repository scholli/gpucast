/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : matrix4x4.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_MATRIX4X4_HPP
#define GPUCAST_MATH_MATRIX4X4_HPP

// includes, system
#include <iosfwd> // fwd. decl: std::ostream

// includes, project

namespace gpucast { namespace math {

template<typename value_t>
class matrix4x4
{
public:

  typedef value_t             value_type;

public :

  matrix4x4                   ( );
  matrix4x4                   ( matrix4x4 const& );
  explicit matrix4x4          ( value_t const [16]);

  ~matrix4x4                  ( );

  void swap                   ( matrix4x4& );

  matrix4x4&      operator=   ( matrix4x4 const& );

  matrix4x4&      operator*=  ( matrix4x4 const& );
  matrix4x4&      operator*=  ( value_t );
  matrix4x4&      operator/=  ( value_t );

  value_t&        operator[]  ( unsigned i );
  value_t const&  operator[]  ( unsigned i ) const;

  value_t         determinant ( ) const;

  bool            invert      ( );

  void            transpose   ( );

  void            decompose   ( matrix4x4<value_t>& translation,
                                matrix4x4<value_t>& rotation,
                                matrix4x4<value_t>& scale ) const;

  matrix4x4       normalmatrix( ) const;

  void            write       ( std::ostream& os ) const;
  void            read        ( std::istream& is );

private:

  value_t data_[16];
};

template<typename value_t>
matrix4x4<value_t> operator-(matrix4x4<value_t> const&);

template<typename value_t>
matrix4x4<value_t> operator*(matrix4x4<value_t> const&, matrix4x4<value_t> const&);

template<typename value_t>
matrix4x4<value_t> operator*(matrix4x4<value_t> const&, value_t);

template<typename value_t>
matrix4x4<value_t> operator*(value_t, matrix4x4<value_t> const&);

template<typename value_t>
matrix4x4<value_t> operator/(matrix4x4<value_t> const&, value_t);

template<typename value_t,
         typename float4_t>
float4_t operator*(matrix4x4<value_t> const&,
                   float4_t           const& );

template<typename value_t>
matrix4x4<value_t> make_translation(value_t, value_t, value_t);

template<typename value_t>
matrix4x4<value_t> make_scale(value_t, value_t, value_t);

template<typename value_t>
matrix4x4<value_t> make_rotation_x(value_t);

template<typename value_t>
matrix4x4<value_t> make_rotation_y(value_t);

template<typename value_t>
matrix4x4<value_t> make_rotation_z(value_t);

template<typename value_t>
matrix4x4<value_t> inverse(matrix4x4<value_t> const&);

template<typename value_t>
matrix4x4<value_t> transpose(matrix4x4<value_t> const&);

template<typename value_t>
std::ostream& operator<<(std::ostream&, matrix4x4<value_t> const&);

template<typename value_t>
matrix4x4<value_t>
frustum      ( value_t left,    value_t right,
               value_t bottom,  value_t	top,
               value_t nearVal, value_t	farVal);

template<typename value_t>
matrix4x4<value_t>
ortho       ( value_t left,     value_t right,
              value_t bottom,   value_t top,
              value_t nearVal,  value_t farVal );

template<typename value_t>
matrix4x4<value_t>
perspective ( value_t fovy,
              value_t aspect,
              value_t nearVal,
              value_t farVal );

template<typename value_t>
matrix4x4<value_t>
lookat      ( value_t eyeX,     value_t eyeY,     value_t eyeZ,
              value_t centerX,  value_t centerY,  value_t centerZ,
              value_t upX,      value_t upY,      value_t upZ );

// predefined typedefs
typedef matrix4x4<float>          matrix4f;
typedef matrix4x4<double>         matrix4d;

} } // namespace gpucast / namespace math

#include <gpucast/math/matrix4x4_impl.hpp>

#endif // GPUCAST_MATH_MATRIX4X4_HPP
