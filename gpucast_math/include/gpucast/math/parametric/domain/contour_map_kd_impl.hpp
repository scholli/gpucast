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

namespace gpucast { namespace math {


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void 
contour_map_kd<value_t>::contour_cell::print ( std::ostream& os ) const
{
  os << "contour cell: v: " << interval_v << ", u: " << interval_u << ", contours: " << overlapping_segments.size() << " inside: " << inside << std::endl;
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_kd<value_t>::contour_map_kd () 
  : _cells(),
    _root ( nullptr )
{}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_kd<value_t>::~contour_map_kd ()
{
  destroy ( _root );
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void                      
contour_map_kd<value_t>::initialize ()
{
  _cells.clear();

  if ( this->_contour_segments.empty () ) return;

  this->update_bounds();

  std::cout << "Segments for partition : " << std::endl;
  unsigned tmp=0;
  for (auto s : this->_contour_segments)
  {
    std::cout << tmp++ << " : " << s->bbox() << std::endl;
  }

  //_cells.push_back(contour_cell{ {}, 
  //                               {_bounds.min[point_type::u],
  //                                _bounds.max[point_type::u] }, 
  //                               {_bounds.min[point_type::v],
  //                                _bounds.max[point_type::v] }, 
  //                               { false } });

  for (auto const& s : this->_contour_segments)
  {
    this->_insert_contour_segment(s);
  }

  //_cells = S.result;

  for (auto const& c : _cells)
  {
    std::cout << "cell : " << std::endl;
    std::cout << " u : " << c.interval_u << std::endl;
    std::cout << " v : " << c.interval_v << std::endl;
  }

  // TODO: O(n^2)! improve performance by sorting
  for ( contour_cell& c : _cells )
  {
    for ( contour_segment_ptr const& p : this->_contour_segments ) 
    {
      interval_type segment_v ( p->bbox().min[point_type::v], p->bbox().max[point_type::v],  gpucast::math::excluded,  gpucast::math::excluded );
      interval_type segment_u ( p->bbox().min[point_type::u], p->bbox().max[point_type::u],  gpucast::math::excluded,  gpucast::math::excluded );
      if ( c.interval_v.overlap ( segment_v ) && 
           c.interval_u.overlap ( segment_u ) )
      {
        c.overlapping_segments.push_back(p);
      }
      unsigned intersections = 0;
      if ( segment_v.in ( c.interval_v.center() ) &&
           c.interval_u.center() < segment_u.minimum() )
      {
        ++intersections;
      }
      c.inside = (intersections%2 == 1);
    }
  }

  // create kd-tree
  _root = create ( this->_bounds, _cells );
 
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void                              
contour_map_kd<value_t>::destroy ( kdnode* n )
{
  if ( n == nullptr ) {
    return;
  } else {

    if ( n->is_child() ) {
      delete n->cell;
      delete n;
    } else {
      destroy ( n->less );
      destroy ( n->more );
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_kd<value_t>::kdnode*                           
contour_map_kd<value_t>::create ( bbox_type const& bounds, 
                                  std::vector<contour_cell> const& cells )
{
  std::set<value_type> usplit = split_candidates ( bounds, point_type::u, cells );
  std::set<value_type> vsplit = split_candidates ( bounds, point_type::v, cells );

  if ( usplit.empty() && vsplit.empty() ) 
  {
    //assert ( cells.size() == 1 );
    if (cells.empty()) {
      return new kdnode(0, 0, nullptr, nullptr, nullptr);
    } else {
      return new kdnode(0, 0, new contour_cell(cells.front()), nullptr, nullptr);
    }
  } else {
    if ( usplit.size() > vsplit.size() )
    {
      return split ( bounds, point_type::u, usplit, cells );
    } else {
      return split ( bounds, point_type::v, vsplit, cells );
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_map_kd<value_t>::kdnode*                           
contour_map_kd<value_t>::split ( bbox_type const& bounds, 
                                 typename point_type::coordinate_type const& dim, 
                                 std::set<value_type> const& candidates,
                                 std::vector<contour_cell> const& cells )
{
  assert ( !candidates.empty() );

  // choose random split plane
  std::size_t random_offset = std::rand()%candidates.size();
  auto split_iter           = candidates.begin();
  std::advance ( split_iter, random_offset );
  value_type split = *split_iter;

  // sort cells
  std::vector<contour_cell> less_cells;
  std::vector<contour_cell> more_cells;

  for ( auto const& c : cells)
  {
    switch ( dim )
    {
      case point_type::u :
        if ( c.interval_u.in ( split ) )
        {
          less_cells.push_back ( c );
          more_cells.push_back ( c );
        }
        if ( c.interval_u.minimum() >= split )
        {
          more_cells.push_back ( c );
        }
        if ( c.interval_u.maximum() <= split )
        {
          less_cells.push_back ( c );
        }
        break;
      case point_type::v :
        if ( c.interval_v.in ( split ) )
        {
          less_cells.push_back ( c );
          more_cells.push_back ( c );
        }
        if ( c.interval_v.minimum() >= split )
        {
          more_cells.push_back ( c );
        }
        if ( c.interval_v.maximum() <= split )
        {
          less_cells.push_back ( c );
        }
        break;
    }
  }

  bbox_type less_bounds = ( dim == point_type::u ) 
                            ? bbox_type ( point_type ( bounds.min[point_type::u], bounds.min[point_type::v] ),
                                          point_type ( split,                     bounds.max[point_type::v] ) )
                            : bbox_type ( point_type ( bounds.min[point_type::u], bounds.min[point_type::v] ),
                                          point_type ( bounds.max[point_type::u], split                     ) );

  bbox_type more_bounds = ( dim == point_type::u ) 
                            ? bbox_type ( point_type ( split,                     bounds.min[point_type::v] ),
                                          point_type ( bounds.max[point_type::u], bounds.max[point_type::v] ) )
                            : bbox_type ( point_type ( bounds.min[point_type::u], split ),
                                          point_type ( bounds.max[point_type::u], bounds.max[point_type::v] ) );

  //std::cout << "split " << split << std::endl;
  //std::cout << "less bbox : " << less_bounds << std::endl;
  //std::cout << "more bbox : " << more_bounds << std::endl;
  //
  //std::cout << "less cells : " << std::endl;
  //for (auto const& c : less_cells)
  //{
  //  c.print(std::cout);
  //}
  //
  //std::cout << "more cells : " << std::endl;
  //for (auto const& c : more_cells)
  //{
  //  c.print(std::cout);
  //}

  return new kdnode ( split, 
                      dim, 
                      nullptr, 
                      create ( less_bounds, less_cells ),
                      create ( more_bounds, more_cells ) );
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::set<value_t>              
contour_map_kd<value_t>::split_candidates ( bbox_type const& bounds, 
                                            typename point_type::coordinate_type const& dim, 
                                            std::vector<contour_cell> const& cells )
{
  std::set<value_type> splits;
  for ( auto const& c : cells )
  {
    switch ( dim ) 
    {
      case point_type::u : 
        splits.insert ( c.interval_u.minimum());
        splits.insert ( c.interval_u.maximum());
        break;           
      case point_type::v :
        splits.insert ( c.interval_v.minimum());
        splits.insert ( c.interval_v.maximum());
        break;
    };
  }

  splits.erase(bounds.min[dim]);
  splits.erase(bounds.max[dim]);

  return splits;
}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void                      
contour_map_kd<value_t>::print ( std::ostream& os ) const
{
  base_type::print(os);
  os << "cells : " << _cells.size() << std::endl;
  for ( contour_cell const& cell : _cells )
  {
    os << "interval u of cell : " << cell.interval_u << std::endl;
    os << "interval v of cell : " << cell.interval_v << std::endl;
    os << "number of segments in cell : " << cell.overlapping_segments.size() << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_kd<value_t>::_insert_contour_segment(contour_segment_ptr const& s)
{}


///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os,  gpucast::math::contour_map_kd<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}

} } // namespace gpucast / namespace math
