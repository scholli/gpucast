/********************************************************************************
*
* Copyright (C) 2015 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : kdsplit_sah.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_KDSPLIT_SAH_HPP
#define GPUCAST_KDSPLIT_SAH_HPP

// includes, system

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_strategy.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
struct kdsplit_sah : public kdsplit_strategy<value_t> {

  /////////////////////////////////////////////////////////////////////////////
  // typedefs
  /////////////////////////////////////////////////////////////////////////////
  typedef point<value_t, 2>                     point_type;
  typedef typename point_type::coordinate_type  coordinate_type;
  typedef kdnode2d<value_t>                     kdnode_t;
  typedef std::shared_ptr<kdnode_t>             kdnode_ptr;
  typedef kdtree2d<value_t>                     kdtree_t;

  struct split_candidate {
    coordinate_type direction;
    value_t         split_value;
    value_t         costs;
  };

  /////////////////////////////////////////////////////////////////////////////
  bool generate(kdtree2d<value_t> const& initial_tree) const override {
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
             coordinate_type const& direction, 
             value_t const& value) const {

    node->split(direction, value);

    // split children
    try_split(node->child_less);
    try_split(node->child_greater);
  }

  /////////////////////////////////////////////////////////////////////////////
  void try_split (kdnode_ptr const& node) const {
    if (!node) return;

    if (node->is_leaf()) {
      auto bbox_size = node->bbox.max - node->bbox.min;

      // find split candidates per direction
      auto splits_candidates_u = node->split_candidates(point_type::u);
      auto splits_candidates_v = node->split_candidates(point_type::v);

      // gather split candidates in single container to compute costs
      std::vector<split_candidate> candidates;
      
      for (auto const& value : splits_candidates_u) {
        candidates.push_back({point_type::u, value, 0.0});
      }

      for (auto const& value : splits_candidates_v) {
        candidates.push_back({ point_type::v, value, 0.0 });
      }
      
      // find and accumulate areas of adjacent bboxes
      for (auto& candidate : candidates) {
        value_t adjacent_area = 0;

        // virtual split at candidate
        node->split(candidate.direction, candidate.split_value);

        // accumulate propabilities
        value_t area_parent  = node->bbox.size().abs();
        value_t area_less    = node->child_less->bbox.size().abs();
        value_t area_greater = node->child_greater->bbox.size().abs();

        value_t propability_less = area_less / area_parent;
        value_t propability_greater = area_greater / area_parent;

        // estimate costs
        value_t costs_less = 0;
        value_t costs_greater = 0;

        for (auto const& s : node->child_less->overlapping_segments) {
          costs_less += s->bbox().size().abs();
        }
        for (auto const& s : node->child_greater->overlapping_segments) {
          costs_greater += s->bbox().size().abs();
        }

        candidate.costs = propability_less * costs_less + propability_greater * costs_greater;

        // undo potential split
        node->child_less.reset();
        node->child_greater.reset();
      }
       
      std::sort(candidates.begin(), candidates.end(), []
        (split_candidate const& lhs, split_candidate const& rhs) { 
        return lhs.costs < rhs.costs;
      });

      if (!candidates.empty()) {
        split(node, candidates.front().direction, candidates.front().split_value);
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

#endif // GPUCAST_KDSPLIT_SAH_HPP
