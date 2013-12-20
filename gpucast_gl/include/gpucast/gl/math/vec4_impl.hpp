/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : vec4_impl.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include <limits>

namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>::vec4()
{
  _data[0] = 0;
  _data[1] = 0;
  _data[2] = 0;
  _data[3] = 0;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>::vec4( value_t x,
                            value_t y,
                            value_t z,
                            value_t w)
{
  _data[0] = x;
  _data[1] = y;
  _data[2] = z;
  _data[3] = w;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename indexed_array_t>
inline vec4<value_t>::vec4( indexed_array_t const& a)
{
  _data[0] = value_t(a[0]);
  _data[1] = value_t(a[1]);
  _data[2] = value_t(a[2]);
  _data[3] = value_t(a[3]);
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>::vec4(vec4<value_t> const& rhs)
{
  _data[0] = rhs._data[0];
  _data[1] = rhs._data[1];
  _data[2] = rhs._data[2];
  _data[3] = rhs._data[3];
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename iterator_t>
inline vec4<value_t>::vec4( iterator_t beg, iterator_t end )
{
  std::copy(beg, end, _data);
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>::~vec4()
{}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t const&
vec4<value_t>::operator[](unsigned c) const
{
  return _data[c];
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t&
vec4<value_t>::operator[](unsigned c)
{
  return _data[c];
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>&
vec4<value_t>::operator=(vec4<value_t> const& rhs)
{
  vec4<value_t> tmp(rhs);
  swap(tmp);
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename indexed_array_t>
inline vec4<value_t>&        
vec4<value_t>::operator=( indexed_array_t const& rhs )
{
  vec4<value_t> tmp(rhs);
  swap(tmp);
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
vec4<value_t>::operator*=(value_t scalar)
{
  _data[0] *= scalar;
  _data[1] *= scalar;
  _data[2] *= scalar;
  _data[3] *= scalar;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
vec4<value_t>::operator/=(value_t scalar)
{
  _data[0] /= scalar;
  _data[1] /= scalar;
  _data[2] /= scalar;
  _data[3] /= scalar;
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
vec4<value_t>::operator-=(vec4<value_t> const& rhs)
{
  _data[0] -= rhs._data[0];
  _data[1] -= rhs._data[1];
  _data[2] -= rhs._data[2];
  _data[3] -= rhs._data[3];
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
vec4<value_t>::operator+=(vec4<value_t> const& rhs)
{
  _data[0] += rhs._data[0];
  _data[1] += rhs._data[1];
  _data[2] += rhs._data[2];
  _data[3] += rhs._data[3];
}



////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline void
vec4<value_t>::swap(vec4& rhs)
{
  std::swap(_data[0], rhs._data[0]);
  std::swap(_data[1], rhs._data[1]);
  std::swap(_data[2], rhs._data[2]);
  std::swap(_data[3], rhs._data[3]);
}



////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline void
vec4<value_t>::print(std::ostream& os) const
{
  os << "[ " << _data[0] << " " << _data[1] << " " << _data[2] << " " << _data[3] << "]";
}



////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
vec4<value_t>::as_homogenous() const
{
  return vec4<value_t>(_data[0] * _data[3],
		                   _data[1] * _data[3],
                       _data[2] * _data[3],
		                   _data[3] );
}



////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
vec4<value_t>::as_euclidian() const
{
  return vec4<value_t>(_data[0] / _data[3],
		                   _data[1] / _data[3],
                       _data[2] / _data[3],
		                   _data[3] );
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::abs() const
{
  return length();
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::length() const
{
  return std::sqrt(_data[0]*_data[0] +
                   _data[1]*_data[1] +
                   _data[2]*_data[2] +
                   _data[3]*_data[3]);
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::length_square() const
{
  return _data[0]*_data[0] +
         _data[1]*_data[1] +
         _data[2]*_data[2] +
         _data[3]*_data[3];
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline void
vec4<value_t>::normalize()
{
  value_t l = length();
  _data[0] /= l;
  _data[1] /= l;
  _data[2] /= l;
  _data[3] /= l;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::x() const
{
  return _data[0];
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::y() const
{
  return _data[1];
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::z() const
{
  return _data[2];
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t
vec4<value_t>::w() const
{
  return _data[3];
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec2<value_t>
vec4<value_t>::xy() const
{
  return vec2<value_t>(_data[0], _data[1]);
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec3<value_t>
vec4<value_t>::xyz() const
{
  return vec3<value_t>(_data[0], _data[1], _data[2]);
}

////////////////////////////////////////////////////////////////////////////////
// static methods
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
/* static */ inline vec4<value_t>
vec4<value_t>::maximum()
{
  return vec4<value_t> (std::numeric_limits<value_t>::max(),
                        std::numeric_limits<value_t>::max(),
                        std::numeric_limits<value_t>::max(),
                        std::numeric_limits<value_t>::max());
}
////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
/* static */ inline vec4<value_t>
vec4<value_t>::minimum()
{
  return vec4<value_t> (std::numeric_limits<value_t>::min(),
                        std::numeric_limits<value_t>::min(),
                        std::numeric_limits<value_t>::min(),
                        std::numeric_limits<value_t>::min());
}


////////////////////////////////////////////////////////////////////////////////
// external methods
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline bool
operator==(vec4<value_t> const& lhs, vec4<value_t> const& rhs)
{
  return (lhs[0] == rhs[0]) &&
         (lhs[1] == rhs[1]) &&
         (lhs[2] == rhs[2]) &&
         (lhs[3] == rhs[3]);
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline bool
operator!=(vec4<value_t> const& lhs, vec4<value_t> const& rhs)
{
  return !(lhs == rhs);
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
operator-(vec4<value_t> const& lhs, vec4<value_t> const& rhs)
{
  vec4<value_t> tmp(lhs);
  tmp -= rhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
elementwise_min(vec4<value_t> const& a, vec4<value_t> const& b)
{
  return vec4<value_t>( std::min(a[0], b[0]),
                        std::min(a[1], b[1]),
                        std::min(a[2], b[2]),
                        std::min(a[3], a[3]));
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
elementwise_max(vec4<value_t> const& a, vec4<value_t> const& b)
{
  return vec4<value_t>( std::max(a[0], b[0]),
                        std::max(a[1], b[1]),
                        std::max(a[2], b[2]),
                        std::max(a[3], b[3]));
}

////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline std::ostream&
operator<<(std::ostream& os, vec4<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
operator*(vec4<value_t> const& lhs, value_t const& rhs)
{
  vec4<value_t> tmp(lhs);
  tmp *= rhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
operator*(value_t const& lhs, vec4<value_t> const& rhs)
{
  vec4<value_t> tmp(rhs);
  tmp *= lhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
operator/(vec4<value_t> const& lhs, value_t const& rhs)
{
  vec4<value_t> tmp(lhs);
  tmp /= rhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline vec4<value_t>
operator+(vec4<value_t> const& lhs, vec4<value_t> const& rhs)
{
  vec4<value_t> tmp(lhs);
  tmp += rhs;
  return tmp;
}


////////////////////////////////////////////////////////////////////////////////
template <typename value_t>
inline value_t 
dot(vec4<value_t> const& lhs, vec4<value_t> const& rhs) 
{
  value_t d(0);
  for (unsigned i = 0; i != vec4<value_t>::size; ++i) 
  {
    d += lhs[i] * rhs[i];
  }
  return d;
}

} } // namespace gpucast / namespace gl
