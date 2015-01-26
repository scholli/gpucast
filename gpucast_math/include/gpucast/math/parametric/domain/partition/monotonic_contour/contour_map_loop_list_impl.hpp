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
  for (auto const& segmented_loop : _segmented_loops) {
    unclassified_loops.push_back(segmented_loop.first);
  }

  // determine outer loop
  auto outer_contour = _determine_outer_loop(unclassified_loops);
  unclassified_loops.erase(std::remove(unclassified_loops.begin(), unclassified_loops.end(), outer_contour), unclassified_loops.end());

  _outer_loop = trimloop{ outer_contour, {} };

  // todo: allow trimloops of depth > 2
  while (!unclassified_loops.empty())
  {
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
contour_map_loop_list<value_t>::trimloop const& root() const {
  return _outer_loop; 
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

  return !is_in_other && parent->is_inside(*child);
}

    } // namespace domain
  } // namespace math
} // namespace gpucast 
