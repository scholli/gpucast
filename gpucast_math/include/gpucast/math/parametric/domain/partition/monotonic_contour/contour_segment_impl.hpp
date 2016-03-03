/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_segment_impl.hpp
*
*  description:
*
********************************************************************************/

namespace gpucast {
  namespace math {
    namespace domain {

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename curve_ptr_iterator_t>
contour_segment<value_t>::contour_segment ( curve_ptr_iterator_t begin,
                                            curve_ptr_iterator_t end )
 : _curves   ( begin, end ),
   _bbox     (),
   _monotony (unclassified),
   _continous(false)
{
  _determine_monotony();
  _continous = is_continous();
  _update_bbox();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_segment<value_t>::~contour_segment()
{}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
monotony_t
contour_segment<value_t>::monotony () const
{
  return _monotony;
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_segment<value_t>::bbox_type const&
contour_segment<value_t>::bbox () const
{
  return _bbox;
}

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool 
contour_segment<value_t>::is_monotonic(typename point_type::coordinate_type const& c) const
{
  if (_curves.empty()) {
    return true;
  }

  // find monotony
  bool found_increasing = false;
  bool found_decreasing = false;

  for (auto const& curve : _curves)
  {
    if (!curve->is_monotonic(c)) {
      return false;
    }

    if (!curve->is_constant(c)) {
      found_increasing |= curve->is_increasing(c);
      found_decreasing |= !curve->is_increasing(c);
    }

    if (found_increasing && found_decreasing) {
      return false;
    }
  }

  // no change in monotony or direction
  return true ;
}

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool
contour_segment<value_t>::is_continous() const
{
  if ( _curves.empty() )
  {
    return false;
  } else {
    if ( _curves.size() > 1 )
    {
      auto fst = _curves.begin();
      auto snd = _curves.begin();
      std::advance ( snd, 1 );
      while ( snd != _curves.end() )
      {
        if ( (**fst).front() != (**snd).back() &&
             (**fst).back()  != (**snd).front() )
        {
          return false;
        }
        ++snd;
        ++fst;
      }
      return true;
    } else {
      return true;
    }
  }
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool contour_segment<value_t>::is_increasing ( typename point_type::coordinate_type const& c ) const
{
  for ( auto curve : _curves )
  {
    if ( !curve->is_increasing ( c ) && !curve->is_constant(c) )
    {
      return false;
    }
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool
contour_segment<value_t>::is_constant ( typename point_type::coordinate_type const& c) const
{
  for ( auto curve = _curves.begin(); curve != _curves.end(); ++curve )
  {
    if ( !(**curve).is_constant ( c ) )
    {
      return false;
    }
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool                  
contour_segment<value_t>::right_of(point_type const& origin) const
{
  auto contour_segment_bbox = bbox();

  // origin in contour's v-interval
  if (origin[point_type::v] >= contour_segment_bbox.min[point_type::v] &&
      origin[point_type::v] <= contour_segment_bbox.max[point_type::v])
  {
    unsigned intersections_on_right = 0;

    for (auto const& curve : _curves)
    {
      auto curve_bbox = curve->bbox_simple();

      // origin in curve's v-interval --> try to intersect
      if (origin[point_type::v] >= curve_bbox.min[point_type::v] &&
          origin[point_type::v] <= curve_bbox.max[point_type::v] )
      {
        bool curve_on_right = 0;
        std::size_t tmp = 0;

        curve->optbisect(origin, point_type::v, point_type::u, curve_on_right, tmp);

        if (curve_on_right)
        {
          ++intersections_on_right;
        }
      }
    }

    // monotonic_segment can have only one intersection
    return intersections_on_right == 1;

  } else {
    return false;
  } 
}

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_segment<value_t>::point_type            
contour_segment<value_t>::intersect(typename point_type::coordinate_type const& direction, value_type const& v) const
{
  try {

    for (auto const& curve : _curves)
    {
      contour_segment<value_t>::bbox_type curve_bbox;
      curve->bbox_simple(curve_bbox);

      // origin in curve's v-interval --> try to intersect
      if (v >= curve_bbox.min[direction] &&
        v <= curve_bbox.max[direction])
      {
        bool is_root = false;
        value_type t = 0;

        curve->bisect(point_type::v, v, is_root, t);

        if (is_root)
        {
          return curve->evaluate(t);
        }
        else {
          throw std::runtime_error("contour_segment<value_t>::intersect(): no intersection");
        }
      }
    }
    throw std::runtime_error("contour_segment<value_t>::intersect(): no curves to intersect");
  }
  catch (std::exception& e) {
    std::cout << "Warning: contour_segment<value_t>::intersect(): No intersection determined." << std::endl;
    return contour_segment<value_t>::point_type{};
  }
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::size_t
contour_segment<value_t>::size () const
{
  return _curves.size();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_segment<value_t>::const_curve_iterator
contour_segment<value_t>::begin () const
{
  return _curves.begin();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour_segment<value_t>::const_curve_iterator
contour_segment<value_t>::end () const
{
  return _curves.end();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_segment<value_t>::clip_horizontal ()
{
  curve_iterator c = _curves.begin();
  while ( c != _curves.end() )
  {
    if ( (*c)->is_constant ( point_type::v ) )
    {
      c = _curves.erase ( c );
    } else {
      ++c;
    }
  }

  _update_bbox();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_segment<value_t>::invert ()
{
  std::reverse ( _curves.begin(), _curves.end() );
  std::for_each ( _curves.begin(), _curves.end(), std::bind ( &curve_type::invert, std::placeholders::_1 ) );
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_segment<value_t>::print ( std::ostream& os ) const
{
  os << "contour segment consisting of " << _curves.size() << " curves : " << std::endl;
  for ( auto c = _curves.begin(); c != _curves.end(); ++c )
  {
    (**c).print(os);
  }
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour_segment<value_t>::_determine_monotony ()
{
  if (_curves.empty()) {
    _monotony = unclassified;
    return;
  }
  else {
    bool u_mono = true;
    bool v_mono = true;

    bool u_increasing = _curves.front()->is_increasing(point_type::u);
    bool v_increasing = _curves.front()->is_increasing(point_type::v);

    for (auto const& curve : _curves)
    {
      if (u_increasing != curve->is_increasing(point_type::u)) {
        u_mono = false;
      }
      if (v_increasing != curve->is_increasing(point_type::v)) {
        v_mono = false;
      }
    }

    if (u_mono && v_mono)   _monotony = bi_monotonic;
    if (u_mono && !v_mono)  _monotony = u_monotonic;
    if (!u_mono && v_mono)  _monotony = v_monotonic;
    if (!u_mono && !v_mono) _monotony = unclassified;
  }
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour_segment<value_t>::_update_bbox ()
{
  if ( _curves.empty() ) return;

  _curves.front()->bbox_simple(_bbox);

  for ( auto c = _curves.begin(); c != _curves.end(); ++c )
  {
     gpucast::math::axis_aligned_boundingbox<point_type> bbox;
    (**c).bbox_simple(bbox);
    _bbox.merge(bbox);
  }
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os,  contour_segment<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}


    } // namespace domain
  } // namespace math
} // namespace gpucast 

