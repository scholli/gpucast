/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_base_impl.hpp
*
*  description:
*
********************************************************************************/
// includes, system
#include <vector>
#include <algorithm>

// includes, project

#define GPUCAST_DEBUG_OPTIMIZATION_STRATEGY 0

namespace gpucast {
  namespace math {
    namespace domain {

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_base<value_t>::contour_map_base ()
: _contour_segments (),
  _segmented_loops(),
  _bounds           ()
{}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_base<value_t>::~contour_map_base()
{}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_base<value_t>::add ( contour_type const& loop )
{
  // keep loop copy as shared_ptr
  auto loop_ptr = std::make_shared<contour_type>(loop);

  // split loop into bimonotnic segments
  contour_segment_container loop_segments = loop_ptr->monotonic_segments();

  // make monotonic segments point into positive v direction
  for (auto const& segment : loop_segments) {
    if (!segment->is_increasing(point_type::v)) {
      segment->invert();
    }
#if GPUCAST_DEBUG_OPTIMIZATION_STRATEGY
    if (!segment->is_monotonic(point_type::u) || !segment->is_monotonic(point_type::v)) {
      std::cout << "Caution: non-bi-monotonic segment!" << std::endl;

      for (auto c = segment->begin(); c != segment->end(); ++c) {
        std::cout << **c << std::endl;
      }
    }
#endif
  }

  // split into bi-monotonic contour segments
  std::copy(loop_segments.begin(), loop_segments.end(), std::back_inserter(_contour_segments));

  // store original loop and its segments
  _segmented_loops[loop_ptr] = loop_segments;

  // update boundary
  update_bounds();
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour_map_base<value_t>::clear ()
{
  _contour_segments.clear();
  update_bounds();
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_base<value_t>::contour_segment_map const&
contour_map_base<value_t>::loops() const
{
  return _segmented_loops;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_base<value_t>::contour_segment_container const& 
contour_map_base<value_t>::monotonic_segments () const
{
  return _contour_segments;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::size_t
contour_map_base<value_t>::count_curves () const
{
  std::size_t ncurves = 0;
  std::for_each ( _contour_segments.begin(), _contour_segments.end(), [&] ( contour_type const& c ) { ncurves += c.curves.size(); } );
  return ncurves;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::vector<typename contour_map_base<value_t>::curve_ptr>
contour_map_base<value_t>::curves () const
{
  std::vector<curve_ptr> curve_container;
  std::for_each ( _contour_segments.begin(), _contour_segments.end(), [&] ( contour_type const& c ) { std::copy ( c.curves.begin(), c.curves.end(), std::back_inserter(curve_container)); } );
  return curve_container;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_base<value_t>::bbox_type const&                  
contour_map_base<value_t>::bounds () const
{
  return _bounds;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour_map_base<value_t>::update_bounds ()
{
  if ( _contour_segments.empty() )
  {
    _bounds = bbox_type();
  } else {
    _bounds = (*_contour_segments.begin())->bbox();
    for ( contour_segment_ptr const& c : _contour_segments )
    {
      _bounds.merge(c->bbox());
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_base<value_t>::print ( std::ostream& os ) const
{
  os << "contour_map:\n";
  os << "monotonic curves in contour_map : " << std::endl;

  std::for_each(_contour_segments.begin(), _contour_segments.end(), std::bind(&contour_segment_type::print, std::placeholders::_1, std::ref(os)));
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void                           
contour_map_base<value_t>::minimize_overlaps()
{
  auto optimized_segments = _contour_segments;
  bool continue_splitting = true;

  while (continue_splitting) {

    // compute current overlaps
    auto costs_before_split = compute_costs(optimized_segments);
    std::vector<overlap> overlaps = find_overlaps(optimized_segments);

    if (overlaps.empty()) {
      continue_splitting = false; // no overlaps found -> no split necessary
    }
    else {
      // try to split segment with biggest overlap
      overlap next_splittable_segment;
      bool splittable_segment_found = get_splittable_segment(overlaps, next_splittable_segment);

      if (!splittable_segment_found) {
        continue_splitting = false; // no splittable segment found
      } else {
#if GPUCAST_DEBUG_OPTIMIZATION_STRATEGY
        std::cout << "choosen split segment : " << next_splittable_segment.segment << " curves = " << next_splittable_segment.segment->size() << std::endl;
#endif
        auto overlaps_before_split = accumulate_overlap_area(overlaps);

        std::map<std::size_t, double> split_candidate_map;
        for (std::size_t i = 0; i != next_splittable_segment.segment->size(); ++i)
        {
          auto tmp_segments = optimized_segments;

          apply_split(tmp_segments, next_splittable_segment.segment, i);

          // TODO: find better benefit function
          auto costs_after_split = compute_costs(tmp_segments);
          if (costs_after_split < costs_before_split) {
            split_candidate_map.insert(std::make_pair(i, costs_after_split));
          }
        }
#if GPUCAST_DEBUG_OPTIMIZATION_STRATEGY
        for (auto p : split_candidate_map) {
          std::cout << "split candidate = " << p.first << " costs = " << p.second << std::endl;
        }
#endif
        if (split_candidate_map.empty())
        {
          continue_splitting = false;
        }
        else {

          auto best_split = get_best_candidate(split_candidate_map);
#if GPUCAST_DEBUG_OPTIMIZATION_STRATEGY
          std::cout << " best_split: " << next_splittable_segment.segment << " pos = " << best_split  << " : " << optimized_segments.size() << std::endl;
#endif
          // apply split
          apply_split(optimized_segments, next_splittable_segment.segment, best_split);
        }
      }
    }
  }
  _contour_segments = optimized_segments;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::vector<typename contour_map_base<value_t>::overlap> 
contour_map_base<value_t>::find_overlaps(contour_segment_container const& segments) const
{
  std::vector<overlap> overlaps;

  // determine overlaps 
  for (auto const& s0 : segments) 
  {
    overlap segment_overlaps = { s0, {}, 0 };

    // test all other segments for overlaps
    for (auto const& s1 : segments) {
      if (s0 != s1) {

        auto b0 = s0->bbox();
        auto b1 = s1->bbox();
        auto bbox = b0 && b1;
        auto overlap_area = bbox.volume();
        
        // if overlap, store
        if (overlap_area > 0) {
          segment_overlaps.other.push_back(s1);
          segment_overlaps.area += overlap_area;
        }
      }
    }
    // add all overlapping segments for this one
    if (segment_overlaps.area > 0) {
      overlaps.push_back(segment_overlaps);
    }
  }

  // sort by overlapping area
  std::sort(overlaps.begin(), overlaps.end(), [](overlap const& lhs, overlap const& rhs) {
    return lhs.area > rhs.area;
  });
#if GPUCAST_DEBUG_OPTIMIZATION_STRATEGY
  for (auto const& o : overlaps) {
    std::cout << "overlap : " << o.segment << " curves:  " << o.segment->size() << " area =" << o.area << std::endl;
  }
#endif
  return overlaps;
};
///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
double contour_map_base<value_t>::accumulate_overlap_area(std::vector<overlap> const& v) const {

  double area = 0;
  for (auto const& o : v) {
    area += o.area;
  }

  return area;
};
///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::size_t contour_map_base<value_t>::get_best_candidate(std::map<std::size_t, double> const& candidates) const 
{

  std::multimap<double, std::size_t> sorted_candidates;
  std::transform(candidates.begin(), candidates.end(), std::inserter(sorted_candidates,
    sorted_candidates.begin()), [](std::pair<std::size_t, double> p) {
    return std::make_pair(p.second, p.first);
  });

  return sorted_candidates.begin()->second;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
double contour_map_base<value_t>::compute_costs(contour_segment_container const& segments) const {

  // re-analyse possible split
  auto tmp_overlaps = find_overlaps(segments);
  auto overlaps_after_split = accumulate_overlap_area(tmp_overlaps);

  return overlaps_after_split;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool contour_map_base<value_t>::get_splittable_segment(std::vector<overlap> const& overlaps, overlap& found) const {

  for (auto o : overlaps) 
  {
    if (o.segment->size() > 1) {
      found = o;
      return true;
    }
  }

  return false;
  
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour_map_base<value_t>::apply_split(contour_segment_container& segments, contour_segment_ptr const& s, std::size_t pos) const
{
  auto splitted_segments = s->split(pos);
  segments.erase(std::remove(segments.begin(), segments.end(), s), segments.end());
  segments.push_back(splitted_segments.first);
  segments.push_back(splitted_segments.second);
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour_map_base<value_t>::optimize_traversal_costs()
{
  // TODO : not fully implemented yet
  for (auto const& s : _contour_segments) {
    value_t area_total = s->bbox().volume(); 
    value_t area_curves = 0;

    for (auto c = s->begin(); c != s->end(); ++c) {
      area_curves += (**c).bbox_simple().volume();
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
/*static*/ void                
contour_map_base<value_t>::_determine_splits ( std::set<value_type>&                       result, 
                                               typename point_type::coordinate_type const& dimension,
                                               contour_segment_container const&            contours )
{
  for ( contour_segment_ptr const& c : contours )
  {
    bbox_type cbbox = c->bbox();
    result.insert(cbbox.min[dimension]);
    result.insert(cbbox.max[dimension]);
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
/*static*/ void
contour_map_base<value_t>::_intervals_from_splits (std::set<value_type> const& input, std::set<interval_type>& output )
{
  if ( input.size() >= 2 )
  {
    auto first  = input.begin();
    auto second = input.begin();
    std::advance(second, 1);
    while ( second != input.end() )
    {
      output.insert( interval_type ( *first, *second, excluded, excluded ) );
      ++first; ++second;
    }
  } 
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void                
contour_map_base<value_t>::_contours_in_interval ( interval_type const& interval,
                                                   typename point_type::coordinate_type const& dimension,
                                                   contour_segment_container const&            input,
                                                   contour_segment_container&                  output ) const
{
  for ( auto c : input ) 
  {
    if ( c->bbox().extends(dimension, excluded, excluded).overlap(interval) )
    {
      output.push_back ( c );
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::size_t         
contour_map_base<value_t>::_contours_greater ( value_type const& value,
                                               typename point_type::coordinate_type const& dimension,
                                               contour_segment_container const& input ) const
{
  std::size_t intersections = 0;
  for ( contour_segment_ptr const& c : input )
  {
    if ( c->bbox().min[dimension] > value )
    {
      ++intersections;
    }
  }
  return intersections;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os, contour_map_base<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}


    } // namespace domain
  } // namespace math
} // namespace gpucast 
