/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : pair_adaptor.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_PAIR_ADAPTOR_HPP
#define GPUCAST_MATH_PAIR_ADAPTOR_HPP

namespace gpucast { namespace math {
  namespace util {

    template <typename pair_type>  
    typename pair_type::second_type const&
    second_adaptor(pair_type const& pair)
    {
      return pair.second;
    }

    template <typename pair_type>  
    typename pair_type::first_type const&
    first_adaptor(pair_type const& pair)
    {
      return pair.first;
    }

  } // namespace util
} } // namespace gpucast / namespace math

#endif
