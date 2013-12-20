/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : erase_if.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_ERASE_IF_HPP
#define GPUCAST_MATH_ERASE_IF_HPP

// header, system

namespace gpucast { namespace math {
  namespace util {

    template <typename container_type, typename pred>
    void erase_if(container_type& c, pred f)
    {
      typedef typename container_type::iterator iterator;

      iterator x = std::find_if(c.begin(), c.end(), f);
      while ( x != c.end() ) 
      {
        c.erase(x);
        x = std::find_if(c.begin(), c.end(), f);
      }
    }

  } // namespace util
} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_ERASE_IF_HPP
