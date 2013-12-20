/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : space_adapter.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_SPACE_ADAPTER_HPP
#define GPUCAST_MATH_SPACE_ADAPTER_HPP

namespace gpucast { namespace math {

      template <typename point_t>
      class point_to_euclid_space_adapter
      {
      public:
        point_t operator()(point_t const& p) const
        {
          return p.as_euclidian();
        }
      };

      template <typename point_t>
      class point_to_homogenous_space_adapter
      {
      public:
        point_t operator()(point_t const& p) const
        {
          return p.as_homogenous();
        }
      };

} } // namespace gpucast / namespace math
#endif
