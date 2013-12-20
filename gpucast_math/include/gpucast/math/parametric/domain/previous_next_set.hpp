/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : previous_next_set.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_PREVIOUS_NEXT_SET
#define GPUCAST_MATH_PREVIOUS_NEXT_SET

namespace gpucast { namespace math {

template <typename ptr_type>
struct previous_next_set
{
  previous_next_set()
    : _prev()
  {}

  void operator()(ptr_type p)
  {
    if (_prev) 
    {
      _prev->next(p);
      p->previous(_prev);
    }
    _prev = p;
  }

  ptr_type _prev;
};

} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_PREVIOUS_NEXT_SET
