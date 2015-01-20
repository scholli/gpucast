/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_binary_impl.hpp
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
contour_map_binary<value_t>::contour_cell::print ( std::ostream& os ) const
{
  os << "contour cell: v: " << interval_v << ", u: " << interval_u << ", contours: " << overlapping_segments.size() << " inside: " << inside << std::endl;
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_binary<value_t>::contour_map_binary ()
  : _cells()
{}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_binary<value_t>::~contour_map_binary ()
{}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_binary<value_t>::initialize ()
{
  _cells.clear();

  std::for_each ( this->_contour_segments.begin(), this->_contour_segments.end(), std::bind ( &base_type::contour_segment_type::clip_horizontal, std::placeholders::_1 ) );

  std::set<typename base_type::value_type>    vsplits;
  contour_map_base<value_t>::_determine_splits ( vsplits, point_type::v, this->_contour_segments );

  std::set<interval_type> vpartition;
  contour_map_base<value_t>::_intervals_from_splits ( vsplits, vpartition );

  for ( interval_type const& v_interval : vpartition )
  {
    contour_segment_container segments_in_v_interval;
    for ( contour_segment_ptr const& c : this->_contour_segments )
    {
      if ( c->bbox().extends ( base_type::point_type::v, excluded, excluded ).overlap ( v_interval ) )
      {
        segments_in_v_interval.push_back ( c );
      }
    }

    std::set<value_type> usplits;
    contour_map_base<value_t>::_determine_splits ( usplits, point_type::u, segments_in_v_interval );

    std::set<interval_type> upartition;
    contour_map_base<value_t>::_intervals_from_splits ( usplits, upartition );

    contour_interval cells_in_v_interval;
    value_type umin = usplits.empty() ? 0 : *usplits.begin();
    value_type umax = usplits.empty() ? 0 : *usplits.rbegin();
    cells_in_v_interval.interval_u = interval_type ( umin, umax );
    cells_in_v_interval.interval_v = v_interval;

    for ( interval_type const& u_interval : upartition )
    {
      contour_cell cell;

      contour_map_base<value_t>::_contours_in_interval ( u_interval, point_type::u, segments_in_v_interval, cell.overlapping_segments );

      std::size_t intersections = contour_map_base<value_t>::_contours_greater ( u_interval.center(), point_type::u, segments_in_v_interval );
      cell.inside     = (intersections%2 == 1);
      cell.interval_u = u_interval;
      cell.interval_v = v_interval;

      cells_in_v_interval.cells.push_back ( cell );
    }
    _cells.push_back ( cells_in_v_interval );
  }
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_binary<value_t>::cell_partition const&
contour_map_binary<value_t>::partition () const
{
  return _cells;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_binary<value_t>::print ( std::ostream& os ) const
{
  base_type::print(os);
  os << "vintervals : " << _cells.size() << std::endl;
  for ( contour_interval const& interval : _cells )
  {
    os << "number of uintervals in vinterval : " << interval.cells.size() << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os,  gpucast::math::contour_map_binary<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}

    } // namespace domain
  } // namespace math
} // namespace gpucast 
