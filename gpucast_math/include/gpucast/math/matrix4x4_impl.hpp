/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : matrix4x4_impl.hpp
*  project    : glpp
*  description:
*
********************************************************************************/

// include i/f header

// includes, system
#include <algorithm>              // std::copy, std::swap_ranges
#include <cmath>                  // std::cos, std::sin
#include <iomanip>                // std::fixed, std::setprecision
#include <ostream>                // std::ostream

// includes, project
#include <gpucast/math/vec3.hpp>

namespace gpucast { namespace math {

// functions, internal

template<typename value_t>
value_t
det3_helper(value_t a1, value_t a2, value_t a3,
	          value_t b1, value_t b2, value_t b3,
	          value_t c1, value_t c2, value_t c3)
{
  return ((a1 * b2 * c3) + (a2 * b3 * c1) + (a3 * b1 * c2) -
          (a1 * b3 * c2) - (a2 * b1 * c3) - (a3 * b2 * c1));
}

// functions, exported
///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>::matrix4x4()
{
  data_[0] = 1.0;
  data_[1] = 0.0;
  data_[2] = 0.0;
  data_[3] = 0.0;

  data_[4] = 0.0;
  data_[5] = 1.0;
  data_[6] = 0.0;
  data_[7] = 0.0;

  data_[8] = 0.0;
  data_[9] = 0.0;
  data_[10] = 1.0;
  data_[11] = 0.0;

  data_[12] = 0.0;
  data_[13] = 0.0;
  data_[14] = 0.0;
  data_[15] = 1.0;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>::matrix4x4(matrix4x4 const& rhs)

{
  std::copy(rhs.data_, rhs.data_+16, data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>::matrix4x4(value_t const a[16])
{
  std::copy(a, a+16, data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>::~matrix4x4()
{}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix4x4<value_t>::swap(matrix4x4& rhs)
{
  std::swap_ranges(data_, data_+16, rhs.data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>&
matrix4x4<value_t>::operator=(matrix4x4 const& rhs)
{
  matrix4x4 tmp(rhs);

  swap(tmp);

  return *this;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>&
matrix4x4<value_t>::operator*=(matrix4x4 const& rhs)
{
  matrix4x4 tmp;

  tmp.data_[ 0] = (data_[ 0] * rhs.data_[ 0] +
		               data_[ 4] * rhs.data_[ 1] +
		               data_[ 8] * rhs.data_[ 2] +
		               data_[12] * rhs.data_[ 3]);
  tmp.data_[ 1] = (data_[ 1] * rhs.data_[ 0] +
		               data_[ 5] * rhs.data_[ 1] +
		               data_[ 9] * rhs.data_[ 2] +
		               data_[13] * rhs.data_[ 3]);
  tmp.data_[ 2] = (data_[ 2] * rhs.data_[ 0] +
		               data_[ 6] * rhs.data_[ 1] +
		               data_[10] * rhs.data_[ 2] +
		               data_[14] * rhs.data_[ 3]);
  tmp.data_[ 3] = (data_[ 3] * rhs.data_[ 0] +
		               data_[ 7] * rhs.data_[ 1] +
		               data_[11] * rhs.data_[ 2] +
		               data_[15] * rhs.data_[ 3]);
  tmp.data_[ 4] = (data_[ 0] * rhs.data_[ 4] +
		               data_[ 4] * rhs.data_[ 5] +
		               data_[ 8] * rhs.data_[ 6] +
		               data_[12] * rhs.data_[ 7]);
  tmp.data_[ 5] = (data_[ 1] * rhs.data_[ 4] +
		               data_[ 5] * rhs.data_[ 5] +
		               data_[ 9] * rhs.data_[ 6] +
		               data_[13] * rhs.data_[ 7]);
  tmp.data_[ 6] = (data_[ 2] * rhs.data_[ 4] +
		               data_[ 6] * rhs.data_[ 5] +
		               data_[10] * rhs.data_[ 6] +
		               data_[14] * rhs.data_[ 7]);
  tmp.data_[ 7] = (data_[ 3] * rhs.data_[ 4] +
		               data_[ 7] * rhs.data_[ 5] +
		               data_[11] * rhs.data_[ 6] +
		               data_[15] * rhs.data_[ 7]);
  tmp.data_[ 8] = (data_[ 0] * rhs.data_[ 8] +
		               data_[ 4] * rhs.data_[ 9] +
		               data_[ 8] * rhs.data_[10] +
		               data_[12] * rhs.data_[11]);
  tmp.data_[ 9] = (data_[ 1] * rhs.data_[ 8] +
		               data_[ 5] * rhs.data_[ 9] +
		               data_[ 9] * rhs.data_[10] +
		               data_[13] * rhs.data_[11]);
  tmp.data_[10] = (data_[ 2] * rhs.data_[ 8] +
		               data_[ 6] * rhs.data_[ 9] +
		               data_[10] * rhs.data_[10] +
		               data_[14] * rhs.data_[11]);
  tmp.data_[11] = (data_[ 3] * rhs.data_[ 8] +
		               data_[ 7] * rhs.data_[ 9] +
		               data_[11] * rhs.data_[10] +
		               data_[15] * rhs.data_[11]);
  tmp.data_[12] = (data_[ 0] * rhs.data_[12] +
		               data_[ 4] * rhs.data_[13] +
		               data_[ 8] * rhs.data_[14] +
		               data_[12] * rhs.data_[15]);
  tmp.data_[13] = (data_[ 1] * rhs.data_[12] +
		               data_[ 5] * rhs.data_[13] +
		               data_[ 9] * rhs.data_[14] +
		               data_[13] * rhs.data_[15]);
  tmp.data_[14] = (data_[ 2] * rhs.data_[12] +
		               data_[ 6] * rhs.data_[13] +
		               data_[10] * rhs.data_[14] +
		               data_[14] * rhs.data_[15]);
  tmp.data_[15] = (data_[ 3] * rhs.data_[12] +
		               data_[ 7] * rhs.data_[13] +
		               data_[11] * rhs.data_[14] +
		               data_[15] * rhs.data_[15]);

  swap(tmp);

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>&
matrix4x4<value_t>::operator*=(value_t rhs)
{
  for (unsigned idx = 0; idx < 16; ++idx)
    data_[idx] *= rhs;

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>&
matrix4x4<value_t>::operator/=(value_t rhs)
{
  for (unsigned idx = 0; idx < 16; ++idx)
    data_[idx] /= rhs;

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t&
matrix4x4<value_t>::operator[](unsigned i)
{
  return data_[i];
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t const&
matrix4x4<value_t>::operator[](unsigned i) const
{
  return data_[i];
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t
matrix4x4<value_t>::determinant() const
{
  value_t const& a1(data_[ 0]);
  value_t const& b1(data_[ 4]);
  value_t const& c1(data_[ 8]);
  value_t const& d1(data_[12]);

  value_t const& a2(data_[ 1]);
  value_t const& b2(data_[ 5]);
  value_t const& c2(data_[ 9]);
  value_t const& d2(data_[13]);

  value_t const& a3(data_[ 2]);
  value_t const& b3(data_[ 6]);
  value_t const& c3(data_[10]);
  value_t const& d3(data_[14]);

  value_t const& a4(data_[ 3]);
  value_t const& b4(data_[ 7]);
  value_t const& c4(data_[11]);
  value_t const& d4(data_[15]);

  return (a1 * det3_helper(b2, b3, b4, c2, c3, c4, d2, d3, d4) -
          b1 * det3_helper(a2, a3, a4, c2, c3, c4, d2, d3, d4) +
          c1 * det3_helper(a2, a3, a4, b2, b3, b4, d2, d3, d4) -
          d1 * det3_helper(a2, a3, a4, b2, b3, b4, c2, c3, c4));
  }


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
bool
matrix4x4<value_t>::invert()
{
  bool         result(false);
  value_t const d(determinant());

  if (0.0 != d) {
    value_t const& a1(data_[ 0]);
    value_t const& b1(data_[ 4]);
    value_t const& c1(data_[ 8]);
    value_t const& d1(data_[12]);
    value_t const& a2(data_[ 1]);
    value_t const& b2(data_[ 5]);
    value_t const& c2(data_[ 9]);
    value_t const& d2(data_[13]);
    value_t const& a3(data_[ 2]);
    value_t const& b3(data_[ 6]);
    value_t const& c3(data_[10]);
    value_t const& d3(data_[14]);
    value_t const& a4(data_[ 3]);
    value_t const& b4(data_[ 7]);
    value_t const& c4(data_[11]);
    value_t const& d4(data_[15]);

    value_t const di(1.0f / d);

    matrix4x4 tmp;

    tmp.data_[ 0] =  det3_helper(b2, b3, b4, c2, c3, c4, d2, d3, d4) * di;
    tmp.data_[ 1] = -det3_helper(a2, a3, a4, c2, c3, c4, d2, d3, d4) * di;
    tmp.data_[ 2] =  det3_helper(a2, a3, a4, b2, b3, b4, d2, d3, d4) * di;
    tmp.data_[ 3] = -det3_helper(a2, a3, a4, b2, b3, b4, c2, c3, c4) * di;
    tmp.data_[ 4] = -det3_helper(b1, b3, b4, c1, c3, c4, d1, d3, d4) * di;
    tmp.data_[ 5] =  det3_helper(a1, a3, a4, c1, c3, c4, d1, d3, d4) * di;
    tmp.data_[ 6] = -det3_helper(a1, a3, a4, b1, b3, b4, d1, d3, d4) * di;
    tmp.data_[ 7] =  det3_helper(a1, a3, a4, b1, b3, b4, c1, c3, c4) * di;
    tmp.data_[ 8] =  det3_helper(b1, b2, b4, c1, c2, c4, d1, d2, d4) * di;
    tmp.data_[ 9] = -det3_helper(a1, a2, a4, c1, c2, c4, d1, d2, d4) * di;
    tmp.data_[10] =  det3_helper(a1, a2, a4, b1, b2, b4, d1, d2, d4) * di;
    tmp.data_[11] = -det3_helper(a1, a2, a4, b1, b2, b4, c1, c2, c4) * di;
    tmp.data_[12] = -det3_helper(b1, b2, b3, c1, c2, c3, d1, d2, d3) * di;
    tmp.data_[13] =  det3_helper(a1, a2, a3, c1, c2, c3, d1, d2, d3) * di;
    tmp.data_[14] = -det3_helper(a1, a2, a3, b1, b2, b3, d1, d2, d3) * di;
    tmp.data_[15] =  det3_helper(a1, a2, a3, b1, b2, b3, c1, c2, c3) * di;

    swap(tmp);

    result = true;
  }

  return result;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix4x4<value_t>::transpose()
{
  matrix4x4 tmp(*this);

  // data_[ 0] = tmp.data_[ 0];
  data_[ 1] = tmp.data_[ 4];
  data_[ 2] = tmp.data_[ 8];
  data_[ 3] = tmp.data_[12];
  data_[ 4] = tmp.data_[ 1];
  // data_[ 5] = tmp.data_[ 5];
  data_[ 6] = tmp.data_[ 9];
  data_[ 7] = tmp.data_[13];
  data_[ 8] = tmp.data_[ 2];
  data_[ 9] = tmp.data_[ 6];
  // data_[10] = tmp.data_[10];
  data_[11] = tmp.data_[14];
  data_[12] = tmp.data_[ 3];
  data_[13] = tmp.data_[ 7];
  data_[14] = tmp.data_[11];
  // data_[15] = tmp.data_[15];
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix4x4<value_t>::decompose( matrix4x4<value_t>& translation,
                               matrix4x4<value_t>& rotation,
                               matrix4x4<value_t>& scale ) const
{
  // extract translation
  value_t tx = data_[3];
  value_t ty = data_[7];
  value_t tz = data_[11];

  translation = make_translation(tx, ty, tz);

  // scaling
  value_t sx = std::sqrt(data_[0]*data_[0] + data_[1]*data_[1] + data_[2]*data_[2]);
  value_t sy = std::sqrt(data_[4]*data_[4] + data_[5]*data_[5] + data_[6]*data_[6]);
  value_t sz = std::sqrt(data_[8]*data_[8] + data_[9]*data_[9] + data_[10]*data_[10]);

  scale = make_scale(sx, sy,sz);

  // rotation
  value_t const tmp[] = { data_[0]/sx, data_[1]/sx, data_[2]/sx,  0,
                          data_[4]/sy, data_[5]/sy, data_[6]/sy,  0,
                          data_[8]/sz, data_[9]/sz, data_[10]/sz, 0,
                          0,           0,           0,            1 };

   rotation = matrix4x4<value_t>(tmp);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
matrix4x4<value_t>::normalmatrix( ) const
{
  matrix4x4 m(*this);

  //std::cout << "m : " << m << std::endl;
  m.invert();
  //std::cout << "m-1 : " << m << std::endl;
  m.transpose();
  //std::cout << "mt : " << m << std::endl;
  return m;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void            
matrix4x4<value_t>::write ( std::ostream& os ) const
{
  os.write ( reinterpret_cast<char const*> (data_), sizeof(data_));
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void           
matrix4x4<value_t>::read ( std::istream& is )
{
  is.read ( reinterpret_cast<char*> (data_), sizeof(data_));
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// external functions and operators
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
operator-(matrix4x4<value_t> const& rhs)
{
  return matrix4x4<value_t>(rhs) *= -1.0;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
operator*(matrix4x4<value_t> const& lhs, matrix4x4<value_t> const& rhs)
{
  return matrix4x4<value_t>(lhs) *= rhs;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
operator*(matrix4x4<value_t> const& lhs, value_t rhs)
{
  return matrix4x4<value_t>(lhs) *= rhs;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
operator*(value_t lhs, matrix4x4<value_t> const& rhs)
{
  return matrix4x4<value_t>(rhs) *= lhs;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
operator/(matrix4x4<value_t> const& lhs, value_t rhs)
{
  return matrix4x4<value_t>(lhs) /= rhs;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t,
          typename float4_t>
float4_t
operator*(matrix4x4<value_t> const&   lhs,
          float4_t const&             rhs)
{
  return float4_t(lhs[0] * rhs[0] + lhs[4] * rhs[1] + lhs[8] * rhs[2] + lhs[12] * rhs[3],
		              lhs[1] * rhs[0] + lhs[5] * rhs[1] + lhs[9] * rhs[2] + lhs[13] * rhs[3],
		              lhs[2] * rhs[0] + lhs[6] * rhs[1] + lhs[10] * rhs[2] + lhs[14] * rhs[3],
		              lhs[3] * rhs[0] + lhs[7] * rhs[1] + lhs[11] * rhs[2] + lhs[15] * rhs[3]);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
make_translation(value_t a, value_t b, value_t c)
{
  matrix4x4<value_t> tmp;

  tmp[12] = a;
  tmp[13] = b;
  tmp[14] = c;

  return tmp;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
make_scale(value_t a, value_t b, value_t c)
{
  matrix4x4<value_t> tmp;

  tmp[0]  = a;
  tmp[5]  = b;
  tmp[10] = c;

  return tmp;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
make_rotation_x(value_t a)
{
  value_t const cos_a(std::cos(a));
  value_t const sin_a(std::sin(a));

  matrix4x4<value_t> tmp;

  tmp[5] =  cos_a;
  tmp[9] =  sin_a;
  tmp[6] = -sin_a;
  tmp[10] =  cos_a;

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
make_rotation_y(value_t a)
{
  value_t const cos_a(std::cos(a));
  value_t const sin_a(std::sin(a));

  matrix4x4<value_t> tmp;

  tmp[0] =  cos_a;
  tmp[8] = -sin_a;
  tmp[2] =  sin_a;
  tmp[10] =  cos_a;

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
make_rotation_z(value_t a)
{
  value_t const cos_a(std::cos(a));
  value_t const sin_a(std::sin(a));

  matrix4x4<value_t> tmp;

  tmp[0] =  cos_a;
  tmp[4] =  sin_a;
  tmp[1] = -sin_a;
  tmp[5] =  cos_a;

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
inverse(matrix4x4<value_t> const& a)
{
  matrix4x4<value_t> tmp(a);

  tmp.invert();

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
transpose(matrix4x4<value_t> const& a)
{
  matrix4x4<value_t> tmp(a);

  tmp.transpose();

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
frustum (value_t	left,
         value_t	right,
         value_t	bottom,
         value_t	top,
         value_t	nearVal,
         value_t	farVal)
{
  /*****************************************
  ( 2*near/(r-l)     0           A        0 )
  (     0        2*near/(t-b)    B        0 )
  (     0            0           C        D )
  (     0            0          -1        0 )

  A = (r+l) / (r-l)
  B = (t+b) / (t-b)
  C = (f+n) / (f-n)
  D = 2*f*n / (f-n)
  ******************************************/

  matrix4x4<value_t> m;

  m[0] = (2*nearVal) / (right - left);
  //m[1] = value_type( 0);
  //m[2] = value_type( 0);
  //m[3] = value_type( 0);

  //m[4] = value_type( 0);
  m[5] = (2*nearVal) / (top - bottom);
  //m[6] = value_type( 0);
  //m[7] = value_type( 0);

  m[8] = (right + left) / (right - left);
  m[9] = (top + bottom) / (top - bottom);
  m[10] = (-(farVal + nearVal)) / (farVal - nearVal);
  m[11] = value_t(-1);

  //m[12] = value_type( 0);
  //m[13] = value_type( 0);
  m[14] = (-2 * nearVal * farVal) / (farVal - nearVal);
  m[15] = value_t( 0);

  return m;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
ortho ( value_t left,
        value_t right,
        value_t bottom,
        value_t top,
        value_t nearVal,
        value_t farVal )
{
  value_t tx = - ((right + left) / (right - left));
  value_t ty = - ((top + bottom) / (top - bottom));
  value_t tz = - ((farVal + nearVal) / (farVal - nearVal));

  matrix4x4<value_t> m;
  m[0] = value_t(2) / (right - left);
  //m[1] = 0;
  //m[2] = 0;
  //m[3] = 0;


  //m[4] = 0;
   m[5] = value_t(2) / (top - bottom);
  //m[6] = 0;
  //m[7] = 0;

  //m[8] = 0;
  //m[9] = 0;
  m[10] = value_t(-2) / (farVal - nearVal);
  //m[11] = 0;

  m[12] = tx;
  m[13] = ty;
  m[14] = tz;
  m[15] = 1;

  return m;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
perspective ( value_t fovy,
              value_t aspect,
              value_t nearVal,
              value_t farVal )
{
  matrix4x4<value_t> m;

  value_t f = std::atan(fovy/2);

  m[0] = f / aspect;
  //m[1] = 0;
  //m[2] = 0;
  //m[3] = 0;

  //m[4] = 0;
  m[5] = f;
  //m[6] = 0;
  //m[7] = 0;

  //m[8] = 0;
  //m[9] = 0;
  m[10] = (nearVal + farVal) / (nearVal - farVal);;
  m[11] = value_t(-1);

  //m[12] = 0;
  //m[13] = 0;
  m[14] = (2 * nearVal * farVal) / (nearVal - farVal);
  m[15] = 0;

  return m;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix4x4<value_t>
lookat( value_t eyeX,
        value_t eyeY,
        value_t eyeZ,
        value_t centerX,
        value_t centerY,
        value_t centerZ,
        value_t upX,
        value_t upY,
        value_t upZ)
{
  typedef vec3<value_t> vec3_type;

  vec3_type f(centerX - eyeX, centerY - eyeY, centerZ - eyeZ);
  f.normalize();

  vec3_type up(upX, upY, upZ);
  up.normalize();

  vec3_type s = cross(f, up);
  vec3_type u = cross(s, f);

  matrix4x4<value_t>  m;

  m[0] = s[0];
  m[1] = u[0];
  m[2] = -f[0];
  m[3] = 0.0;

  m[4] = s[1];
  m[5] = u[1];
  m[6] = -f[1];
  m[7] = 0.0;

  m[8] = s[2];
  m[9] = u[2];
  m[10] = -f[2];
  m[11] = 0.0;

  //m[12] = 0;
  //m[13] = 0;
  //m[14] = 0;
  //m[15] = 1;

  m = make_translation(-eyeX, -eyeY, -eyeZ) * m;

  return m;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
std::ostream&
operator<<(std::ostream& os, matrix4x4<value_t> const& a)
{
  std::ostream::sentry const cerberus(os);

  if (cerberus) {
    os << std::fixed << std::setprecision(3)
       << '['
       << a[0] << ','
       << a[4] << ','
       << a[8] << ','
       << a[12] << ','
       << std::endl
       << ' '
       << a[1] << ','
       << a[5] << ','
       << a[9] << ','
       << a[13] << ','
       << std::endl
       << ' '
       << a[2] << ','
       << a[6] << ','
       << a[10] << ','
       << a[14] << ','
       << std::endl
       << ' '
       << a[3] << ','
       << a[7] << ','
       << a[11] << ','
       << a[15]
       << ']';
  }

  return os;
}

} } // namespace gpucast / namespace math
