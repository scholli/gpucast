/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : copy_if.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_COPY_IF_HPP
#define GPUCAST_MATH_COPY_IF_HPP

namespace gpucast { namespace math {
  namespace util {

    template <typename input_iterator, typename output_iterator, typename pred>
    void copy_if(input_iterator beg, input_iterator end, output_iterator beg2, pred p)
    {
      while (beg != end) 
      {
        if (p(*beg))
        {
          *beg2++ = *beg;
        }
        ++beg;
      }
    }

  } // namespace util
} } // namespace gpucast / namespace math

#endif
