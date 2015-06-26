/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_kd_impl.hpp
*
*  description:
*
********************************************************************************/
// includes, system

// includes, project

namespace gpucast {
  namespace math {
    namespace domain {

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_loop_list<value_t>::initialize()
{
  // extract loops
  std::vector<contour_ptr> unclassified_loops;
  for (auto const& segmented_loop : this->_segmented_loops) {
    unclassified_loops.push_back(segmented_loop.first);
  }

  // 1. determine outer loop
  auto outer_contour = _determine_outer_loop(unclassified_loops);
  unclassified_loops.erase(std::remove(unclassified_loops.begin(), unclassified_loops.end(), outer_contour), unclassified_loops.end());

  _outer_loop = trimloop{ outer_contour, {} };

  // 2. determine other loos: TODO: allow trimloops of depth > 2
  while (!unclassified_loops.empty()) {

    auto const& child_contour = unclassified_loops.front();
    if (_is_child(outer_contour, child_contour, unclassified_loops)) {
      trimloop child { child_contour, {} };
      _outer_loop.children.push_back(child);
    }
    else {
      throw std::runtime_error("contour_map_loop_list<value_t>::initialize(): Nested loops with depth > 2 not considered.");
    }
    unclassified_loops.erase(std::remove(unclassified_loops.begin(), unclassified_loops.end(), child_contour), unclassified_loops.end());
  }

  // determine parity and priority for contour segments
  _determine_contour_segments_parity();

  _determine_contour_segments_priority();
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_loop_list<value_t>::print(std::ostream& os) const
{
  os << "contour_map_kd<value_t>::print() not implemented" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_loop_list<value_t>::trimloop const&
contour_map_loop_list<value_t>::root() const
{
  return _outer_loop;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
unsigned             
contour_map_loop_list<value_t>::parity(contour_segment_ptr const& segment) const
{
  return _segment_parity_classification.at(segment);
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
unsigned                     
contour_map_loop_list<value_t>::priority(contour_segment_ptr const& segment) const
{
  return _segment_priority_classification.at(segment);
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os, gpucast::math::domain::contour_map_loop_list<value_t> const& rhs)
{
  return os;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_base<value_t>::contour_ptr const& 
contour_map_loop_list<value_t>::_determine_outer_loop(contour_container const& in_loops)
{
  for (auto const& loop : in_loops)
  {
    bool is_outer = true;
    for (auto const& other : in_loops)
    {
      if (loop != other) {
        if (other->is_inside(*loop)) {
          is_outer = false;
        }
      }
    }
    if (is_outer) {
      return loop;
    }
  }
  throw std::runtime_error("contour_map_loop_list<value_t>::_determine_outer_loop() : Couldn't determine outer loop.");
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool 
contour_map_loop_list<value_t>::_is_child(contour_ptr const& parent, contour_ptr const& child, contour_container const& other_loops)
{
  bool is_in_other = false;

  for (auto const& other : other_loops)
  {
    if (other->is_inside(*child) && child != other) {
      is_in_other = true;
    }
  }

  // loop is child loop if it is in parent AND OUTSIDE ALL OTHER loops
  return !is_in_other && parent->is_inside(*child);
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_loop_list<value_t>::_determine_contour_segments_parity()
{
  // TODO: get rid of this magic offset
  value_t epsilon_offset = 0.0001f;

  // classifiy segments
  for (auto const& segment_to_classify : this->_contour_segments) {

    auto const& bbox = segment_to_classify->bbox();

    if (!segment_to_classify->is_constant(point_type::v)) 
    {
      point_type ray_origin;

      if (!segment_to_classify->is_constant(point_type::u)) {
        // for normal segment use epsilon offset of intersection point at half of its v-range
        ray_origin = segment_to_classify->intersect(point_type::v, (bbox.min[point_type::v] + bbox.max[point_type::v]) / 2);
        ray_origin[point_type::u] -= epsilon_offset;
      }
      else {
        // for vertical segments use epsilon offset of intersection point at half of its v-range
        ray_origin = point_type(
          segment_to_classify->bbox().min[point_type::u] - epsilon_offset,
          segment_to_classify->bbox().center()[point_type::v]
        );
      }

      unsigned intersections = 0;
      for (auto const& segment_to_test : this->_contour_segments) {

        intersections += segment_to_test->right_of(ray_origin);
      }

      // store pre-computed parity
      _segment_parity_classification.insert(std::make_pair(segment_to_classify, intersections%2==0));
    }
    else {
      // do not count intersections with horizontal segments
      _segment_parity_classification.insert(std::make_pair(segment_to_classify, 0));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_loop_list<value_t>::_determine_contour_segments_priority()
{
  // classifiy segments
  for (auto const& segment_to_classify : this->_contour_segments) {
    unsigned priority = 0;
    auto bbox = segment_to_classify->bbox();

    for (auto const& segment_to_test : this->_contour_segments) {

      if (segment_to_test->bbox().is_inside(bbox.center()) &&
          segment_to_classify->bbox().volume() <= segment_to_test->bbox().volume()) {

        ++priority;
      }

    }
    _segment_priority_classification.insert(std::make_pair(segment_to_classify, priority));
  }
}

    } // namespace domain
  } // namespace math
} // namespace gpucast 
