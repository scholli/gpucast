/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : matrix3x3_impl.hpp
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

namespace gpucast { namespace math {

// functions, internal
template<typename value_t>
value_t det2_helper(value_t a11, value_t a21, 
                    value_t a12, value_t a22) 
{
  return a11*a22 - a12*a21;
}


// functions, exported
///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>::matrix3x3()
{
  data_[0] = 1.0;
  data_[1] = 0.0;
  data_[2] = 0.0;

  data_[3] = 0.0;
  data_[4] = 1.0;
  data_[5] = 0.0;

  data_[6] = 0.0;
  data_[7] = 0.0;
  data_[8] = 1.0;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>::matrix3x3 ( value_t a11, value_t a21, value_t a31,
                                value_t a12, value_t a22, value_t a32,
                                value_t a13, value_t a23, value_t a33)
{
  data_[0] = a11;
  data_[1] = a12;
  data_[2] = a13;

  data_[3] = a21;
  data_[4] = a22;
  data_[5] = a23;

  data_[6] = a31;
  data_[7] = a32;
  data_[8] = a33;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>::matrix3x3(matrix3x3 const& rhs)

{
  std::copy(rhs.data_, rhs.data_+9, data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>::matrix3x3(value_t const a[9])
{
  std::copy(a, a+9, data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>::~matrix3x3()
{}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix3x3<value_t>::swap(matrix3x3& rhs)
{
  std::swap_ranges(data_, data_+9, rhs.data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>&
matrix3x3<value_t>::operator=(matrix3x3 const& rhs)
{
  matrix3x3 tmp(rhs);

  swap(tmp);

  return *this;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>&
matrix3x3<value_t>::operator*=(matrix3x3 const& rhs)
{
  matrix3x3 tmp;

  tmp.data_[ 0] = (data_[ 0] * rhs.data_[ 0] +
		               data_[ 3] * rhs.data_[ 1] +
		               data_[ 6] * rhs.data_[ 2]);
  tmp.data_[ 1] = (data_[ 1] * rhs.data_[ 0] +
		               data_[ 4] * rhs.data_[ 1] +
		               data_[ 7] * rhs.data_[ 2]);
  tmp.data_[ 2] = (data_[ 2] * rhs.data_[ 0] +
		               data_[ 5] * rhs.data_[ 1] +
		               data_[ 8] * rhs.data_[ 2]);

  tmp.data_[ 3] = (data_[ 0] * rhs.data_[ 3] +
		               data_[ 3] * rhs.data_[ 4] +
		               data_[ 6] * rhs.data_[ 5]);
  tmp.data_[ 4] = (data_[ 1] * rhs.data_[ 3] +
		               data_[ 4] * rhs.data_[ 4] +
		               data_[ 7] * rhs.data_[ 5]);
  tmp.data_[ 5] = (data_[ 2] * rhs.data_[ 3] +
		               data_[ 5] * rhs.data_[ 4] +
		               data_[ 8] * rhs.data_[ 5]);

  tmp.data_[ 6] = (data_[ 0] * rhs.data_[ 6] +
		               data_[ 3] * rhs.data_[ 7] +
		               data_[ 6] * rhs.data_[ 8]);
  tmp.data_[ 7] = (data_[ 1] * rhs.data_[ 6] +
		               data_[ 4] * rhs.data_[ 7] +
		               data_[ 7] * rhs.data_[ 8]);
  tmp.data_[ 8] = (data_[ 2] * rhs.data_[ 6] +
		               data_[ 5] * rhs.data_[ 7] +
		               data_[ 8] * rhs.data_[ 8]);

  swap(tmp);

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>&
matrix3x3<value_t>::operator*=(value_t rhs)
{
  for (unsigned idx = 0; idx < 9; ++idx)
    data_[idx] *= rhs;

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>&
matrix3x3<value_t>::operator/=(value_t rhs)
{
  for (unsigned idx = 0; idx < 9; ++idx)
    data_[idx] /= rhs;

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t&
matrix3x3<value_t>::operator[](unsigned i)
{
  return data_[i];
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t const&
matrix3x3<value_t>::operator[](unsigned i) const
{
  return data_[i];
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t> 
matrix3x3<value_t>::adjoint( ) const
{
  matrix3x3<value_t> adj;

  adj[0] =  det2_helper(data_[4], data_[5], data_[7], data_[8]);
  adj[1] = -det2_helper(data_[3], data_[5], data_[6], data_[8]);
  adj[2] =  det2_helper(data_[3], data_[4], data_[6], data_[7]);

  adj[3] = -det2_helper(data_[1], data_[2], data_[7], data_[8]);
  adj[4] =  det2_helper(data_[0], data_[2], data_[6], data_[8]);
  adj[5] = -det2_helper(data_[0], data_[1], data_[6], data_[7]);

  adj[6] =  det2_helper(data_[1], data_[2], data_[4], data_[5]);
  adj[7] = -det2_helper(data_[0], data_[2], data_[3], data_[5]);
  adj[8] =  det2_helper(data_[0], data_[1], data_[3], data_[4]);

  return adj;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t
matrix3x3<value_t>::determinant() const
{
  return value_t (data_[0] * data_[4] * data_[8] +
                  data_[3] * data_[7] * data_[2] +
                  data_[6] * data_[1] * data_[5] -
                  data_[6] * data_[4] * data_[2] -
                  data_[3] * data_[1] * data_[8] -
                  data_[0] * data_[7] * data_[5]);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
bool
matrix3x3<value_t>::invert()
{
  // | a11 a12 a13 |-1             |   a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13  |
  // | a21 a22 a23 |    =  1/DET * | -(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13) |
  // | a31 a32 a33 |               |   a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12  |

  bool          result(false);
  value_t const d(determinant());

  if (0.0 != d) 
  {
    matrix3x3<value_t> tmp;

    tmp[0] =  (data_[8]*data_[4] - data_[5]*data_[7]) / d;
    tmp[1] = -(data_[8]*data_[1] - data_[2]*data_[7]) / d;
    tmp[2] =  (data_[5]*data_[1] - data_[2]*data_[4]) / d;
    
    tmp[3] = -(data_[8]*data_[3] - data_[5]*data_[6]) / d;
    tmp[4] =  (data_[8]*data_[0] - data_[2]*data_[6]) / d;
    tmp[5] = -(data_[5]*data_[0] - data_[2]*data_[3]) / d;

    tmp[6] =  (data_[7]*data_[3] - data_[4]*data_[6]) / d;
    tmp[7] = -(data_[7]*data_[0] - data_[1]*data_[6]) / d;
    tmp[8] =  (data_[4]*data_[0] - data_[1]*data_[3]) / d;

    swap(tmp);

    result = true;
  }

  return result;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix3x3<value_t>::transpose()
{
  matrix3x3 tmp(*this);

  // data_[ 0] = tmp.data_[ 0];
  data_[ 1] = tmp.data_[ 3];
  data_[ 2] = tmp.data_[ 6];

  data_[ 3] = tmp.data_[ 1];
  // data_[ 4] = tmp.data_[ 4];
  data_[ 5] = tmp.data_[ 7];

  data_[ 6] = tmp.data_[ 2];
  data_[ 7] = tmp.data_[ 5];
  // data_[8] = tmp.data_[8];
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>
operator-(matrix3x3<value_t> const& rhs)
{
  return matrix3x3<value_t>(rhs) *= -1.0;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>
operator*(matrix3x3<value_t> const& lhs, matrix3x3<value_t> const& rhs)
{
  return matrix3x3<value_t>(lhs) *= rhs;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>
operator*(matrix3x3<value_t> const& lhs, value_t rhs)
{
  return matrix3x3<value_t>(lhs) *= rhs;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>
operator*(value_t lhs, matrix3x3<value_t> const& rhs)
{
  return matrix3x3<value_t>(rhs) *= lhs;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>
operator/(matrix3x3<value_t> const& lhs, value_t rhs)
{
  return matrix3x3<value_t>(lhs) /= rhs;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t, 
          typename vec3_t>
vec3_t
operator*(matrix3x3<value_t> const&   lhs, 
          vec3_t const&               rhs)
{
  return vec3_t(lhs[0] * rhs[0] + lhs[3] * rhs[1] + lhs[6] * rhs[2], 
		            lhs[1] * rhs[0] + lhs[4] * rhs[1] + lhs[7] * rhs[2], 
		            lhs[2] * rhs[0] + lhs[5] * rhs[1] + lhs[8] * rhs[2]);
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix3x3<value_t>
inverse(matrix3x3<value_t> const& a)
{
  matrix3x3<value_t> tmp(a);

  tmp.invert();

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t, typename vector3_t>
matrix3x3<value_t>
transpose(vector3_t const& a)
{
  matrix3x3<value_t> tmp(a);

  tmp.transpose();

  return tmp;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
std::ostream&
operator<<(std::ostream& os, matrix3x3<value_t> const& a)
{
  std::ostream::sentry const cerberus(os);

  if (cerberus) {
    os << std::fixed << std::setprecision(3)
       << '['
       << a[0] << ','
       << a[3] << ','
       << a[6] << ','
       << std::endl
       << ' '
       << a[1] << ','
       << a[4] << ','
       << a[7] << ','
       << std::endl
       << ' '
       << a[2] << ','
       << a[5] << ','
       << a[8] << ','
       << std::endl
       << ']';
  }

  return os;
}

} } // namespace gpucast / namespace math
