/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_impl.hpp
*
*  description:
*
********************************************************************************/
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_segment.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
template <typename curve_ptr_iterator_t>
contour<value_t>::contour ( curve_ptr_iterator_t begin, curve_ptr_iterator_t end )
: _curves(begin, end)
{
  monotonize();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour<value_t>::~contour()
{}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool contour<value_t>::valid () const
{
  if ( _curves.empty() )
  {
    return false;
  } else {
    curve_ptr const& c0 = _curves.front();
    curve_ptr const& cn = _curves.back();

    // check if curve is closed
    if ( c0->front() != cn->back() )
    {
      return false;
    }

    // iterate all points to check if curve is piecewise continous
    point_type const& current = c0->back();
    for ( auto c : _curves )
    {
      if ( c != c0 )
      {
        if ( current != c->front() )
        {
          return false;
        } else {
          current = c->back();
        }
      }
    }
  }

  // past all checks -> valid
  return true;
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool
contour<value_t>::empty   () const
{
  return _curves.empty();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::size_t
contour<value_t>::size    () const
{
  return _curves.size();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour<value_t>::const_curve_iterator
contour<value_t>::begin() const
{
  return _curves.begin();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour<value_t>::const_curve_iterator
contour<value_t>::end() const
{
  return _curves.end();
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour<value_t>::curve_container const& 
contour<value_t>::curves() const
{
  return _curves;
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
typename contour<value_t>::bbox_type
contour<value_t>::bbox() const
{
  if (_curves.empty()) throw std::runtime_error("contour<value_t>::bbox(): Invalid bbox. No curves.");

  bbox_type bbox;
  _curves.front()->bbox_simple(bbox);

  for (auto const& curve : _curves)
  {
    bbox_type b;
    curve->bbox_simple(b);
    bbox.merge(b);
  }

  return bbox;
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool                  
contour<value_t>::is_inside(point_type const& origin) const
{
  // do point-in-polygon test with monotonic segments
  unsigned intersections_to_right = 0;

  for (auto const& segment : _monotonic_segments) {
    intersections_to_right += unsigned(segment->right_of(origin));
  }

  return intersections_to_right % 2 == 1;
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool                  
contour<value_t>::is_inside(contour const& other) const
{
  assert(!empty());
  auto const& first_curve_ptr = *other.begin();
  return is_inside(first_curve_ptr->front());
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour<value_t>::monotonize ()
{
  std::vector<curve_type> contour_of_bimonotonic_curves;

  // split contour at all extremas
  for ( auto const& c : _curves )
  {
    std::vector<curve_type> u_monotonic_curves = c->monotonize(point_type::u);

    for (auto const& c_um : u_monotonic_curves) {
      auto bi_monotonic = c_um.monotonize(point_type::v);
      std::copy(bi_monotonic.begin(), bi_monotonic.end(), std::back_inserter(contour_of_bimonotonic_curves));
    }
  }

  // trace contour - consisting of bimonotonic curve segments
  std::vector<curve_ptr> current_contour;
  _curves.clear();

  bool contour_restart = true;
  bool u_increasing    = false;
  bool v_increasing    = false;

  for ( curve_type const& curve : contour_of_bimonotonic_curves)
  {
    // store new bi-monotonic curves
    auto c_ptr = std::make_shared<curve_type>(curve);
    _curves.push_back(c_ptr);

    // start first bi-monotonic contour
    if ( contour_restart )
    {
      contour_restart = false;

      u_increasing = c_ptr->is_increasing(point_type::u);
      v_increasing = c_ptr->is_increasing(point_type::v);

      current_contour.push_back(c_ptr);
    } else {
      // if curve has same monotony as contour -> add curve to current contour
      if (u_increasing == c_ptr->is_increasing(point_type::u) &&
          v_increasing == c_ptr->is_increasing(point_type::v) &&
          !c_ptr->is_constant(point_type::u) &&
          !c_ptr->is_constant(point_type::v))
      {
        current_contour.push_back(c_ptr);
      } else { // push previous contour_segment and restart new contour_segment

        if (!current_contour.empty()) {
          auto monotonic_segment = std::make_shared<contour_segment_type>(current_contour.begin(), current_contour.end());
          _monotonic_segments.push_back(monotonic_segment);

          current_contour.clear();
        }

        // restart new contour segment
        u_increasing = c_ptr->is_increasing(point_type::u);
        v_increasing = c_ptr->is_increasing(point_type::v);

        current_contour.push_back(c_ptr);
      }
    }
  }

  // flush last contour
  if ( !current_contour.empty() )
  {
    auto monotonic_segment = std::make_shared<contour_segment_type>(current_contour.begin(), current_contour.end());
    _monotonic_segments.push_back(monotonic_segment);
  }
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::vector<typename contour<value_t>::contour_segment_ptr> const&
contour<value_t>::monotonic_segments() const
{
  return _monotonic_segments;
}

/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void contour<value_t>::print ( std::ostream& os ) const
{
  os << "contour with " << _curves.size() <<  " curves :" << std::endl;
  std::for_each ( _curves.begin(), _curves.end(), std::bind ( &curve_type::print, std::placeholders::_1, std::ref(os)));
}


/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os,  contour<value_t> const& rhs)
{
  rhs.print(os);
  return os;
}


    } // namespace domain
  } // namespace math
} // namespace gpucast 

