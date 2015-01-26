/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : util.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_UTIL_HPP
#define GPUCAST_CORE_UTIL_HPP

#include <gpucast/core/conversion.hpp>

// header, system
#include <algorithm>
#include <iterator>
#include <map>
#include <vector>

#include <limits>


namespace gpucast {

//////////////////////////////////////////////////////////////////////////////
template<typename point_t>
class minimum_coord {
public :
  typedef typename point_t::value_type value_t;

public :
  minimum_coord(enum point_t::coord coord)
    : min_(std::numeric_limits<float>::max()),
      coord_(coord)
  {}

  void operator()(point_t const& p) {
    if (min_ > p[coord_]) min_ = p[coord_];
  }

  value_t result() { return min_; }

private :
  value_t min_;
  enum point_t::coord coord_;
};


//////////////////////////////////////////////////////////////////////////////
template<typename point_t>
class maximum_coord {
public :
  typedef typename point_t::value_type value_t;

public :
  maximum_coord(enum point_t::coord coord)
    : max_    ( std::numeric_limits<float>::min() ),
      coord_  ( coord )
  {}

  void operator()(point_t const& p) {
    if(max_ < p[coord_])max_ = p[coord_];
  }

  value_t result() { return max_; }

private :
  value_t max_;
  enum point_t::coord coord_;
};



////////////////////////////////////////////////////////////////////////////////
namespace numeric_helper {
  template <typename value_t>
  bool
  weak_equals(value_t const& a, value_t const& b, value_t const& tol)
  {
    return ( a < (b + tol) && a > (b - tol));
  }

  template <typename value_t>
  bool
  weak_less(value_t const& a, value_t const& b, value_t const& tol)
  {
    return ( a < (b - tol) );
  }

  template <typename value_t>
  bool
  weak_greater(value_t const& a, value_t const& b, value_t const& tol)
  {
    return ( a > (b + tol) );
  }

} // numeric_helper


////////////////////////////////////////////////////////////////////////////////
template <typename argument_type, typename result_type>
class convert {
public :
  result_type
  operator()(argument_type const& val) const {
    return result_type(val);
  }
};


////////////////////////////////////////////////////////////////////////////////
template <typename pair_t, typename value_t>
class map_sec {
public :
  value_t
  operator()(pair_t const& pair) const {
    return pair.second;
  }
};



////////////////////////////////////////////////////////////////////////////////
// input : two container of different type (a,b)
// function : - convert all elements of container a and insert in container b
//            - return position of first element in target container
////////////////////////////////////////////////////////////////////////////////
template <typename source_container_t, typename target_container_t>
class insert_adapter {
public :
  typedef typename source_container_t::value_type source_t;
  typedef typename target_container_t::value_type target_t;

public :
  insert_adapter(target_container_t& target,
		 std::map<std::size_t, std::size_t>& id_map,
		 std::size_t index = 0)
    : target_(target),
      id_map_(id_map),
      index_(index)
  {}

  void
  operator()(source_container_t const& a) {
    std::size_t pos = target_.size();

    std::transform(a.begin(), a.end(),
		   std::back_inserter(target_),
		   convert<source_t, target_t>());
    id_map_.insert(std::make_pair(index_, pos));
    ++index_;
  }
private:
  target_container_t& target_;
  std::map<std::size_t, std::size_t>& id_map_;
  std::size_t index_;
};

inline unsigned 
uint4x8ToUInt ( unsigned char input0, unsigned char input1, unsigned char input2, unsigned char input3 )
{
  unsigned result = 0U;
  result |= (input3 & 0x000000FF) << 24U;
  result |= (input2 & 0x000000FF) << 16U;
  result |= (input1 & 0x000000FF) << 8U;
  result |= (input0 & 0x000000FF);
  return result;
}

inline unsigned
uint2x16ToUInt(unsigned short input0, unsigned short input1)
{
  unsigned result = 0U;
  result |= (input1 & 0x0000FFFF) << 16U;
  result |= (input0 & 0x0000FFFF);
  return result;
}

inline unsigned 
uint2ToUInt ( unsigned short input0, unsigned short input1 )
{
  unsigned result = 0U;
  result |= (input1 & 0x0000FFFF) << 16U;
  result |= (input0 & 0x0000FFFF);
  return result;
}




} // namespace gpucast

#endif // LIBNURBS_UTIL_HPP
