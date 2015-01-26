/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : uniform_cell.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_UNIFORM_CELL_HPP
#define GPUCAST_MATH_UNIFORM_CELL_HPP

// includes, system
#include <set>

// includes, project
#include <gpucast/math/parametric/domain/partition/double_binary/beziercurve_segment.hpp>
#include <gpucast/math/interval.hpp>
#include <gpucast/math/parametric/point.hpp>

// forward declarations
namespace gpucast {
  namespace math {
    namespace domain {


  template <typename value_t > class vertical_interval;

  template <typename value_type>
  class uniform_cell
  {
    public : // typedefs / enums

      typedef point<value_type,2>                              point_type;

      typedef vertical_interval<value_type>                   vertical_interval_type;
      typedef std::shared_ptr<vertical_interval_type>       vertical_interval_ptr;

      typedef uniform_cell<value_type>                        uniform_cell_type;
      typedef std::shared_ptr<uniform_cell_type>            shared_ptr_type;

      typedef beziercurve_segment<point_type>                 curvesegment_type;
      typedef std::shared_ptr<curvesegment_type>            curvesegment_ptr;
      typedef std::set<curvesegment_ptr>                      curvesegment_ptr_set;

      typedef typename curvesegment_ptr_set::iterator         iterator;
      typedef typename curvesegment_ptr_set::const_iterator   const_iterator;

      typedef  gpucast::math::interval<value_type>                       interval_type;

    public : // c'tor / d'tor

      uniform_cell();

      uniform_cell(vertical_interval_ptr    parent,
                   interval_type const&     horizontal_range,
                   interval_type const&     vertical_range);

      uniform_cell(vertical_interval_ptr const& parent,
                   interval_type const&         horizontal_range,
                   interval_type const&         vertical_range,
                   shared_ptr_type const&       prev,
                   shared_ptr_type const&       next);

      ~uniform_cell();

    public : // methods

      void                    next                    ( shared_ptr_type n );
      void                    previous                ( shared_ptr_type n );

      shared_ptr_type         next                    ( ) const;
      shared_ptr_type         previous                ( ) const;

      void                    intersections           ( std::size_t known_intersections );
      std::size_t             intersections           ( ) const;

      interval_type&          get_horizontal_interval ();
      interval_type&          get_vertical_interval   ();

      void                    add                     ( curvesegment_ptr const& );
      void                    remove                  ( curvesegment_ptr const& );
      value_type              minimal_distance        ( curvesegment_type const& ) const;

      std::size_t             size                    () const;

      // clear segment container and known intersections - keep all other properties
      void                    clear                   ();

      iterator                begin                   ();
      iterator                end                     ();

      const_iterator          begin                   () const;
      const_iterator          end                     () const;

      void                    print                   ( std::ostream& os, std::string const& addinfo = "") const;

    private : // internal methods

    private : // attributes

      vertical_interval_ptr                           _parent;

      interval_type                                   _horizontal_interval;
      interval_type                                   _vertical_interval;
      value_type                                      _area;

      shared_ptr_type                                 _previous;
      shared_ptr_type                                 _next;

      std::size_t                                     _intersections;
      curvesegment_ptr_set                            _curves_near_cell;
  };

template <typename value_type>
std::ostream& operator<<(std::ostream& os, gpucast::math::domain::uniform_cell<value_type> const& cell);

} // namespace domain
} // namespace math
} // namespace gpucast 

#include "uniform_cell_impl.hpp"

#endif // GPUCAST_MATH_UNIFORM_CELL_HPP
