/********************************************************************************
*
* Copyright (C) 2015 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : kdsplit_greedy.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_KDSPLIT_GREEDY_HPP
#define GPUCAST_KDSPLIT_GREEDY_HPP

// includes, system
#include <exception>

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_strategy.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
struct kdsplit_greedy : public kdsplit_strategy<value_t> {

  typedef point<value_t, 2>                     point_type;
  typedef typename point_type::coordinate_type  coordinate_type;
  typedef kdnode2d<value_t>                     kdnode_t;
  typedef std::shared_ptr<kdnode_t>             kdnode_ptr;
  typedef kdtree2d<value_t>                     kdtree_t;

  /////////////////////////////////////////////////////////////////////////////
  // typedefs
  /////////////////////////////////////////////////////////////////////////////

  bool generate(kdtree2d<value_t> const& initial_tree) const override {
    #if WIN32
      throw std::exception("Not implemented");
    #else
      return false;
    #endif
  }

  kdnode_ptr find_splittable_node(kdnode_ptr const& node) const {
    if (node->is_leaf()) {
      if (!node->split_candidates(point_type::u).empty() ||
          !node->split_candidates(point_type::v).empty()) {
        return node;
      }
      else {
        return nullptr;
      }
    }
    else {
      find_splittable_node(node->child_less);
      find_splittable_node(node->child_greater);
    }
  }

};

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#endif // GPUCAST_KDSPLIT_GREEDY_HPP
