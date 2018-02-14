/********************************************************************************
*
* Copyright (C) 2015 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : kdsplit_strategy.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_KDSPLIT_STRATEGY_HPP
#define GPUCAST_KDSPLIT_STRATEGY_HPP

// includes, system

// includes, project
//#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdtree2d.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

template <typename T> struct kdtree2d;

template <typename value_t>
struct kdsplit_strategy {

  /////////////////////////////////////////////////////////////////////////////
  // typedefs
  /////////////////////////////////////////////////////////////////////////////
  virtual bool generate(kdtree2d<value_t> const& initial_tree) const = 0;
};

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#endif // GPUCAST_KDSPLIT_STRATEGY_HPP
