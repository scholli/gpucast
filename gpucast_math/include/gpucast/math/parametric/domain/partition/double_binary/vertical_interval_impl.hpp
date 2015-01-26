/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : vertical_interval_impl.hpp
*
*  description:
*
********************************************************************************/
// includes, system
#include <iomanip>

// includes, project
#include <gpucast/math/interval.hpp>
#include <gpucast/math/parametric/domain/partition/double_binary/uniform_cell.hpp>
#include <gpucast/math/parametric/domain/partition/determine_splits_from_endpoints.hpp>
#include <gpucast/math/parametric/domain/partition/previous_next_set.hpp>

namespace gpucast {
  namespace math {
    namespace domain {


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
vertical_interval<value_type>::vertical_interval( interval_type const&  i,
                                                  partition_ptr_type    parent )
: _interval (i),
  _partition(parent),
  _previous (),
  _next     (),
  _segments (),
  _cell_set (),
  _area     ()
{}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
vertical_interval<value_type>::vertical_interval(interval_type const&  i,
                                                 partition_ptr_type    parent,
                                                 shared_ptr_type       prec,
                                                 shared_ptr_type       succ )
: _interval (i),
  _partition(parent),
  _previous (prec),
  _next     (succ),
  _segments (),
  _cell_set (),
  _area     ()
{}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
vertical_interval<value_type>::~vertical_interval()
{}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::previous ( shared_ptr_type p )
{
  _previous = p;
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::next ( shared_ptr_type s )
{
  _next = s;
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::shared_ptr_type
vertical_interval<value_type>::previous ( ) const
{
  return _previous;
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::shared_ptr_type
vertical_interval<value_type>::next ( ) const
{
  return _next;
}


template <typename value_type>
interval<value_type> const&
vertical_interval<value_type>::get_vertical_interval ( ) const
{
  return _interval;
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
interval<value_type>
vertical_interval<value_type>::get_horizontal_interval () const
{
  if (_cell_set.empty())
  {
    return interval_type(0,0);
  } else
  {
    return interval_type((*_cell_set.begin())->get_horizontal_interval().minimum(),
                         (*_cell_set.rbegin())->get_horizontal_interval().maximum());
  }
}

///////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::add ( curve_segment_ptr const& curve_segment )
{
  _segments.insert(curve_segment);
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::add ( cell_ptr_type const& cell )
{
  _cell_set.insert(cell);
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
std::size_t
vertical_interval<value_type>::curve_segments () const
{
  return _segments.size();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::print ( std::ostream& os, std::string const& addinfo ) const
{
  os << "vertical interval " << _interval << " includes cells : " << _cell_set.size() << std::endl;
  std::for_each(_cell_set.begin(), _cell_set.end(), std::bind(&cell_type::print, std::placeholders::_1, std::ref(os), "\n"));
  os << "curves : " << std::endl;
  std::for_each(_segments.begin(), _segments.end(), std::bind(&curve_segment_type::print, std::placeholders::_1, std::ref(os), "\n"));
  os << addinfo;
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
std::size_t
vertical_interval<value_type>::size () const
{
  return _cell_set.size();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
bool
vertical_interval<value_type>::empty () const
{
  return _cell_set.empty();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::clear ()
{
  return _cell_set.clear();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::iterator
vertical_interval<value_type>::begin ()
{
  return _cell_set.begin();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::iterator
vertical_interval<value_type>::end ()
{
  return _cell_set.end();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::const_iterator
vertical_interval<value_type>::begin () const
{
  return _cell_set.begin();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::const_iterator
vertical_interval<value_type>::end () const
{
  return _cell_set.end();
}



///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::reverse_iterator
vertical_interval<value_type>::rbegin ()
{
  return _cell_set.rbegin();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::reverse_iterator
vertical_interval<value_type>::rend ()
{
  return _cell_set.rend();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::const_reverse_iterator
vertical_interval<value_type>::rbegin () const
{
  return _cell_set.rbegin();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::const_reverse_iterator
vertical_interval<value_type>::rend () const
{
  return _cell_set.rend();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::cell_ptr_type const&
vertical_interval<value_type>::front () const
{
  if (_cell_set.empty()) {
    throw std::out_of_range("vertical_interval<value_type>::front (). Container empty.");
  } else {
    return *_cell_set.begin();
  }
}

///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::cell_ptr_type const&
vertical_interval<value_type>::back () const
{
  if (_cell_set.empty()) {
    throw std::out_of_range("vertical_interval<value_type>::back (). Container empty.");
  } else {
    return *_cell_set.rbegin();
  }
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::cell_ptr_type
vertical_interval<value_type>::find ( value_type const& value) const
{
  for (cell_ptr_type const& cell : _cell_set)
  {
    if (cell->get_horizontal_interval().in(value))
    {
      return cell;
    }
  }
  return cell_ptr_type();
}

///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::curve_segment_const_iterator
vertical_interval<value_type>::segment_begin () const
{
  return _segments.begin();
}


///////////////////////////////////////////////////////////////////////////
template <typename value_type>
typename vertical_interval<value_type>::curve_segment_const_iterator
vertical_interval<value_type>::segment_end () const
{
  return _segments.end();
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_type>
void
vertical_interval<value_type>::update_vicinity () const
{
  std::for_each(_cell_set.begin(), _cell_set.end(), previous_next_set<cell_ptr_type>());
}



///////////////////////////////////////////////////////////////////////////
template <typename value_type>
std::ostream& operator<<(std::ostream& os, vertical_interval<value_type> const& rhs)
{
  rhs.print(os);
  return os;
}


    } // namespace domain
  } // namespace math
} // namespace gpucast 


