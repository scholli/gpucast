/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : matrix2x2_impl.hpp
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


// functions, exported
///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>::matrix2x2()
{
  data_[0] = 1.0;
  data_[1] = 0.0;

  data_[2] = 0.0;
  data_[3] = 1.0;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>::matrix2x2 ( value_t a11, value_t a21,
                                value_t a12, value_t a22 )
{
  data_[0] = a11;
  data_[1] = a12;

  data_[2] = a21;
  data_[3] = a22;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>::matrix2x2(matrix2x2 const& rhs)

{
  std::copy(rhs.data_, rhs.data_+4, data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>::matrix2x2(value_t const a[9])
{
  std::copy(a, a+4, data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>::~matrix2x2()
{}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix2x2<value_t>::swap(matrix2x2& rhs)
{
  std::swap_ranges(data_, data_+4, rhs.data_);
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>&
matrix2x2<value_t>::operator=(matrix2x2 const& rhs)
{
  matrix2x2 tmp(rhs);

  swap(tmp);

  return *this;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>&
matrix2x2<value_t>::operator*=(matrix2x2 const& rhs)
{
  matrix2x2 tmp;

  tmp.data_[ 0] = (data_[ 0] * rhs.data_[ 0] +
		               data_[ 2] * rhs.data_[ 1]);
  tmp.data_[ 1] = (data_[ 1] * rhs.data_[ 0] +
		               data_[ 3] * rhs.data_[ 1]);
  tmp.data_[ 2] = (data_[ 0] * rhs.data_[ 2] +
		               data_[ 2] * rhs.data_[ 3]);
  tmp.data_[ 3] = (data_[ 1] * rhs.data_[ 2] +
		               data_[ 3] * rhs.data_[ 3]);

  swap(tmp);

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>&
matrix2x2<value_t>::operator*=(value_t rhs)
{
  for (unsigned idx = 0; idx < 4; ++idx)
    data_[idx] *= rhs;

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>&
matrix2x2<value_t>::operator/=(value_t rhs)
{
  for (unsigned idx = 0; idx < 4; ++idx)
    data_[idx] /= rhs;

  return *this;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t&
matrix2x2<value_t>::operator[](unsigned i)
{
  return data_[i];
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t const&
matrix2x2<value_t>::operator[](unsigned i) const
{
  return data_[i];
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t> 
matrix2x2<value_t>::adjoint( ) const
{
  matrix2x2<value_t> adj;

  adj[0] =  data_[3];
  adj[1] = -data_[2];

  adj[2] = -data_[1];
  adj[3] =  data_[0];

  return adj;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
value_t
matrix2x2<value_t>::determinant() const
{
  return data_[0]*data_[3] - data_[1]*data_[2];
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
bool
matrix2x2<value_t>::invert()
{
  // | a11 a12 a13 |-1             |   a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13  |
  // | a21 a22 a23 |    =  1/DET * | -(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13) |
  // | a31 a32 a33 |               |   a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12  |

  bool          result(false);
  value_t const d(determinant());

  if (0.0 != d) 
  {
    matrix2x2<value_t> tmp;

    tmp[0] =  (data_[3]) / d;
    tmp[1] = -(data_[1]) / d;

    tmp[2] = -(data_[2]) / d;
    tmp[3] =  (data_[0]) / d;

    swap(tmp);

    result = true;
  }

  return result;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
void
matrix2x2<value_t>::transpose()
{
  matrix2x2 tmp(*this);

  //data_[ 0] = tmp.data_[ 0];
  data_[ 1] = tmp.data_[ 2];
  data_[ 2] = tmp.data_[ 1];
  //data_[ 3] = tmp.data_[ 1];

}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>
operator-(matrix2x2<value_t> const& rhs)
{
  return matrix2x2<value_t>(rhs) *= -1.0;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>
operator*(matrix2x2<value_t> const& lhs, matrix2x2<value_t> const& rhs)
{
  return matrix2x2<value_t>(lhs) *= rhs;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>
operator*(matrix2x2<value_t> const& lhs, value_t rhs)
{
  return matrix2x2<value_t>(lhs) *= rhs;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>
operator*(value_t lhs, matrix2x2<value_t> const& rhs)
{
  return matrix2x2<value_t>(rhs) *= lhs;
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>
operator/(matrix2x2<value_t> const& lhs, value_t rhs)
{
  return matrix2x2<value_t>(lhs) /= rhs;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t, 
          typename vec2_t>
vec2_t
operator*(matrix2x2<value_t> const&   lhs, 
          vec2_t const&               rhs)
{
  return vec2_t(lhs[0] * rhs[0] + lhs[2] * rhs[1], 
		            lhs[1] * rhs[0] + lhs[3] * rhs[1]);
}

///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
matrix2x2<value_t>
inverse(matrix2x2<value_t> const& a)
{
  matrix2x2<value_t> tmp(a);

  tmp.invert();

  return tmp;
}


///////////////////////////////////////////////////////////////////////////////
template<typename value_t>
std::ostream&
operator<<(std::ostream& os, matrix2x2<value_t> const& a)
{
  std::ostream::sentry const cerberus(os);

  if (cerberus) {
    os << std::fixed << std::setprecision(3)
       << '['
       << a[0] << ','
       << a[2] << ','
       << std::endl
       << ' '
       << a[1] << ','
       << a[3] << ','
       << std::endl
       << ']';
  }

  return os;
}

} } // namespace gpucast / namespace math
