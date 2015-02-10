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

// includes, project

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
    if (!segment->is_monotonic(point_type::u) && !segment->is_monotonic(point_type::v)) {
      std::cout << "Caution: non-bi-monotonic segment!" << std::endl;
    }
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
