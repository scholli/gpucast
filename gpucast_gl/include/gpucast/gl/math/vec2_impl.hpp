/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vec2_impl.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include <boost/numeric/conversion/bounds.hpp>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>::vec2()
{
  _data[0] = 0;
  _data[1] = 0;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>::vec2( value_t x,
                            value_t y)
{
  _data[0] = x;
  _data[1] = y;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename indexed_array_t>
inline vec2<value_t>::vec2( indexed_array_t const& a)
{
  _data[0] = a[0];
  _data[1] = a[1];
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>::vec2(vec2<value_t> const& rhs)
{
  _data[0] = rhs._data[0];
  _data[1] = rhs._data[1];
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>::~vec2()
{}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t const&
vec2<value_t>::operator[](unsigned c) const
{
  return _data[c];
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t&
vec2<value_t>::operator[](unsigned c)
{
  return _data[c];
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>&
vec2<value_t>::operator=(vec2<value_t> const& rhs)
{
  vec2<value_t> tmp(rhs);
  swap(tmp);
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename indexed_array_t>
inline vec2<value_t>&
vec2<value_t>::operator=( indexed_array_t const& rhs )
{
  vec2<value_t> tmp(rhs);
  swap(tmp);
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>&
vec2<value_t>::operator*=(value_t scalar)
{
  _data[0] *= scalar;
  _data[1] *= scalar;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>&
vec2<value_t>::operator/=(value_t scalar)
{
  _data[0] /= scalar;
  _data[1] /= scalar;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>&
vec2<value_t>::operator-=(vec2<value_t> const& rhs)
{
  _data[0] -= rhs._data[0];
  _data[1] -= rhs._data[1];
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>&
vec2<value_t>::operator+=(vec2<value_t> const& rhs)
{
  _data[0] += rhs._data[0];
  _data[1] += rhs._data[1];
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline void
vec2<value_t>::swap(vec2& rhs)
{
  std::swap(_data[0], rhs._data[0]);
  std::swap(_data[1], rhs._data[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline void
vec2<value_t>::print(std::ostream& os) const
{
  os << "[ " << _data[0] << " " << _data[1] << "]";
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
vec2<value_t>::as_homogenous() const
{
  return vec2( _data[0] * _data[1],
		           _data[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
vec2<value_t>::as_euclidian() const
{
  return vec2( _data[0] / _data[1],
		           _data[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec2<value_t>::abs() const
{
  return length();
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec2<value_t>::length() const
{
  return std::sqrt(_data[0]*_data[0] +
                   _data[1]*_data[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec2<value_t>::length_square() const
{
  return _data[0]*_data[0] + _data[1]*_data[1];
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline void
vec2<value_t>::normalize()
{
  value_t l = length();
  _data[0] /= l;
  _data[1] /= l;
}




////////////////////////////////////////////////////////////////////////////////
// static methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
/* static */ inline vec2<value_t>
vec2<value_t>::maximum()
{
  return vec2<value_t> (std::numeric::bounds<value_t>::highest(),
                        std::numeric::bounds<value_t>::highest());
}
////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
/* static */ inline vec2<value_t>
vec2<value_t>::minimum()
{
  return vec2<value_t> (std::numeric::bounds<value_t>::lowest(),
                        std::numeric::bounds<value_t>::lowest());
}






////////////////////////////////////////////////////////////////////////////////
// external methods
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline bool
operator==(vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  return (lhs[0] == rhs[0]) &&
         (lhs[1] == rhs[1]);
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline bool
operator!=(vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  return !(lhs == rhs);
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator/( vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  return vec2<value_t>(lhs[0]/rhs[0],
                       lhs[1]/rhs[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator*( vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  return vec2<value_t>(lhs[0]*rhs[0],
                       lhs[1]*rhs[1]);
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator+( vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  return vec2<value_t>(lhs[0]+rhs[0],
                       lhs[1]+rhs[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator-( vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  return vec2<value_t>(lhs[0]-rhs[0],
                       lhs[1]-rhs[1]);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
elementwise_min(vec2<value_t> const& a, vec2<value_t> const& b)
{
  return vec2<value_t>(std::min(a[0], b[0]),
                       std::min(a[1], b[1]));
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
elementwise_max(vec2<value_t> const& a, vec2<value_t> const& b)
{
  return vec2<value_t>(std::max(a[0], b[0]),
                       std::max(a[1], b[1]));
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
length ( vec2<value_t> const& p)
{
  return p.length();
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline std::ostream&
operator<<(std::ostream& os, vec2<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator*(vec2<value_t> const& lhs, value_t const& rhs)
{
  vec2<value_t> tmp(lhs);
  tmp *= rhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator*(value_t const& lhs, vec2<value_t> const& rhs)
{
  vec2<value_t> tmp(rhs);
  tmp *= lhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
operator/(vec2<value_t> const& lhs, value_t const& rhs)
{
  vec2<value_t> tmp(lhs);
  tmp /= rhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
dot(vec2<value_t> const& lhs, vec2<value_t> const& rhs)
{
  value_t d(0);
  for (unsigned i = 0; i != vec2<value_t>::size; ++i)
  {
    d += lhs[i] * rhs[i];
  }
  return d;
}


} } // namespace gpucast / namespace gl
