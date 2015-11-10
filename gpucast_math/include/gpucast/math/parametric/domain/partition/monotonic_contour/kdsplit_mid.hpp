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
#ifndef GPUCAST_KDSPLIT_MID_HPP
#define GPUCAST_KDSPLIT_MID_HPP

// includes, system

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_strategy.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
struct kdsplit_mid : public kdsplit_strategy<value_t> {

  /////////////////////////////////////////////////////////////////////////////
  // typedefs
  /////////////////////////////////////////////////////////////////////////////
  typedef point<value_t, 2>                     point_type;
  typedef typename point_type::coordinate_type  coordinate_type;
  typedef kdnode2d<value_t>                     kdnode_t;
  typedef std::shared_ptr<kdnode_t>             kdnode_ptr;
  typedef kdtree2d<value_t>                     kdtree_t;

  /////////////////////////////////////////////////////////////////////////////
  bool generate(typename kdtree2d<value_t> const& initial_tree) const override {
    try {
      try_split(initial_tree.root);
      return true;
    }
    catch (std::exception& e) {
      std::cout << "kdsplit_mid::generate(): exception : " << e.what() << std::endl;
      return false;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void split(kdnode_ptr const& node,
             coordinate_type direction,
             std::set<value_t> const& candidates) const {
    // serialize
    std::vector<value_t> candidates_indexed(candidates.begin(), candidates.end());

    // split along u
    auto mid_id = candidates_indexed.size() / 2;
    node->split(direction, candidates_indexed[mid_id]);

    // split children
    try_split(node->child_less);
    try_split(node->child_greater);
  }

  /////////////////////////////////////////////////////////////////////////////
  void try_split (kdnode_ptr const& node) const {
    if (!node) return;

    if (node->is_leaf()) {
      auto bbox_size = node->bbox.max - node->bbox.min;

      auto splits_candidates_u = node->split_candidates(point_type::u);
      auto splits_candidates_v = node->split_candidates(point_type::v);

      if (bbox_size[point_type::u] > bbox_size[point_type::v]) { // u longer side
        if (!splits_candidates_u.empty()) { // split along u
          split(node, point_type::u, splits_candidates_u);
        }
        else { // try to split v
          if (!splits_candidates_v.empty()) {
            split(node, point_type::v, splits_candidates_v); // split along v
          }
          else {
            // no split possible
          }
        }
      }
      else { // v longer side
        if (!splits_candidates_v.empty()) {
          split(node, point_type::v, splits_candidates_v); // split along v
        }
        else { // try to split u
          if (!splits_candidates_u.empty()) {
            split(node, point_type::u, splits_candidates_u); // split along u
          }
          else {
            // no split possible
          }
        }
      }
    }
    else {
      try_split(node->child_less);
      try_split(node->child_greater);
    }
  }

};

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#endif // GPUCAST_KDSPLIT_MID_HPP
