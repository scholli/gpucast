/********************************************************************************
*
* Copyright (C) 2015 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : kdtree2d.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_KDTREE2D_HPP
#define GPUCAST_KDTREE2D_HPP

// includes, system
#include <memory>

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdnode2d.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_strategy.hpp>

//template <typename T> class kdsplit_strategy;

namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
struct kdtree2d {

  /////////////////////////////////////////////////////////////////////////////
  // typedefs
  /////////////////////////////////////////////////////////////////////////////
  typedef value_t                                  value_type;
  typedef contour_segment<value_type>              contour_segment_type;
  typedef std::shared_ptr<contour_segment_type>    contour_segment_ptr;
  typedef typename contour_segment_type::bbox_type bbox_type;
  typedef std::shared_ptr<kdnode2d<value_type>>    kdnode_ptr;

  /////////////////////////////////////////////////////////////////////////////
  // methods
  /////////////////////////////////////////////////////////////////////////////
  bool initialize(kdsplit_strategy<value_t> const& split_strategy, std::set<contour_segment_ptr> const& segments) {
    if (segments.empty()) {
      root = std::make_shared<kdnode2d<value_t>>(bbox_type(), 0, 0, 0, 0, segments, nullptr, nullptr, nullptr);
      return false;
    }
    else {
      // gather bbox information
      bbox = (*segments.begin())->bbox();
      for (auto const& segment : segments) {
        bbox.merge(segment->bbox());
      }
      // create root
      root = std::make_shared<kdnode2d<value_t>>(bbox, 0, 0, 0, 0, segments, nullptr, nullptr, nullptr);

      bool success = split_strategy.generate(*this);
      root->determine_parity(segments);
      
      return success;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  kdnode_ptr is_in_node(bbox_type const& bb) const {
    return root->is_in_node(bb);
  }

  /////////////////////////////////////////////////////////////////////////////
  value_t traversal_costs() const {
    value_t absolute_costs = root->traversal_costs_absolute();
    return absolute_costs / bbox.size().abs();
  }

  /////////////////////////////////////////////////////////////////////////////
  // member
  /////////////////////////////////////////////////////////////////////////////
  bbox_type                     bbox;
  kdnode_ptr                    root = nullptr;
};

template <typename value_type>
std::ostream& operator<<(std::ostream& os, kdtree2d<value_type> const& rhs);

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#endif // GPUCAST_KDTREE2D_HPP
