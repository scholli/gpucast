/********************************************************************************
*
* Copyright (C) 2015 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : kdnode2d.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_KDNODE2D_HPP
#define GPUCAST_KDNODE2D_HPP

// includes, system

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_base.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
struct kdnode2d : std::enable_shared_from_this<kdnode2d<value_t>>{

  /////////////////////////////////////////////////////////////////////////////
  // typedefs + ctor
  /////////////////////////////////////////////////////////////////////////////
  typedef value_t                               value_type;
  typedef point<value_t, 2>                     point_type;
  typedef typename point_type::coordinate_type  coordinate_type;
  typedef axis_aligned_boundingbox<point_type>  bbox_type;
  typedef contour_segment<value_t>              contour_segment_type;
  typedef std::shared_ptr<contour_segment_type> contour_segment_ptr;
  typedef kdnode2d<value_t>                     kdnode_type;
  typedef std::shared_ptr<kdnode_type>          kdnode_ptr;

  /////////////////////////////////////////////////////////////////////////////
  kdnode2d(bbox_type const& b, 
           value_type v, 
           coordinate_type c, 
           unsigned par,
           std::set<contour_segment_ptr> const& s, 
           kdnode_ptr const& p, 
           kdnode_ptr const& cl,
           kdnode_ptr const& cg) :
    bbox(b),
    split_value(v),
    split_direction(c),
    parity(par),
    overlapping_segments(s),
    parent(p),
    child_less(cl),
    child_greater(cg) 
  {}

  /////////////////////////////////////////////////////////////////////////////
  // methods
  /////////////////////////////////////////////////////////////////////////////
  bool is_leaf() { 
    return child_less == nullptr && 
           child_greater == nullptr; 
  }

  /////////////////////////////////////////////////////////////////////////////
  void serialize_dfs(std::vector<kdnode_ptr>& nodes) {
    nodes.push_back(shared_from_this());
    if (!is_leaf()) {
      child_less->serialize_dfs(nodes);
      child_greater->serialize_dfs(nodes);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void serialize_bfs(std::vector<kdnode_ptr>& nodes) {
    if (parent == nullptr) { // root
      nodes.push_back(shared_from_this());
    } 
    
    if (!is_leaf()) {
      nodes.push_back(child_less);
      nodes.push_back(child_greater);

      child_less->serialize_bfs(nodes);
      child_greater->serialize_bfs(nodes);
    }
    
  }

  /////////////////////////////////////////////////////////////////////////////
  bool split(coordinate_type direction, value_t value) {
    // node not splittable
    if (value >= bbox.max[direction] ||
      value <= bbox.min[direction]) {
      return false;
    }

    // assume split parameters
    split_value = value;
    split_direction = direction;

    // compute children
    bbox_type bbox_less = bbox; 
    bbox_less.max[direction] = value;

    bbox_type bbox_greater = bbox;
    bbox_greater.min[direction] = value;

    // assign segments to children
    std::set<contour_segment_ptr> less_segments;
    std::set<contour_segment_ptr> greater_segments;
    for (auto const& s : overlapping_segments) {
      if (bbox_less.overlap(s->bbox(),false)) {
        less_segments.insert(s);
      }
      if (bbox_greater.overlap(s->bbox(), false)) {
        greater_segments.insert(s);
      }
    }
    
    child_less = std::make_shared<kdnode2d>(bbox_less, 0, 0, 0, less_segments, shared_from_this(), nullptr, nullptr);
    child_greater = std::make_shared<kdnode2d>(bbox_greater, 0, 0, 0, greater_segments, shared_from_this(), nullptr, nullptr);

    overlapping_segments.clear();

    return true;
  }

  /////////////////////////////////////////////////////////////////////////////
  std::set<value_t> split_candidates(coordinate_type direction) const {
    std::set<value_t> split_candidates;
    for (auto const& segment : overlapping_segments) {
      value_t split_candidate0 = segment->bbox().min[direction];
      value_t split_candidate1 = segment->bbox().max[direction];

      if (split_candidate0 > bbox.min[direction] && split_candidate0 < bbox.max[direction]) {
        split_candidates.insert(split_candidate0);
      }

      if (split_candidate1 > bbox.min[direction] && split_candidate1 < bbox.max[direction]) {
        split_candidates.insert(split_candidate1);
      }
    }
    return split_candidates;
  }

  /////////////////////////////////////////////////////////////////////////////
  void determine_parity(std::set<contour_segment_ptr> const& segments) {
    if (is_leaf()) {
      point_type center = (bbox.min + bbox.max) / 2.0;
      std::size_t sum_parity = 0;
      for (auto const& s : segments) {
        if (center[point_type::u] <= s->bbox().min[point_type::u] && // - center is on left side of segments bbox
            center[point_type::v] < s->bbox().max[point_type::v] && // - center in 
            center[point_type::v] > s->bbox().min[point_type::v] && 
            overlapping_segments.count(s) == 0) {
            ++sum_parity;
        }
      }
      parity = sum_parity;
    } else {
      child_less->determine_parity(segments);
      child_greater->determine_parity(segments);
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // member
  /////////////////////////////////////////////////////////////////////////////
  bbox_type                        bbox;

  value_t                          split_value = 0;
  coordinate_type                  split_direction = 0;
  unsigned                         parity = 0;

  std::set<contour_segment_ptr>    overlapping_segments;

  kdnode_ptr                       parent        = nullptr;
  kdnode_ptr                       child_less    = nullptr;
  kdnode_ptr                       child_greater = nullptr;
};

/////////////////////////////////////////////////////////////////////////////
// external functions
/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os, kdnode2d<value_t> const& rhs);

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#endif // GPUCAST_KDNODE2D_HPP
