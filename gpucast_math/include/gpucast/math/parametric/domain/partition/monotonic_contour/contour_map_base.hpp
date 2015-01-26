/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_base.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_CONTOUR_MAP_BASE_HPP
#define GPUCAST_MATH_CONTOUR_MAP_BASE_HPP

// includes, system
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_segment.hpp>

// includes, project
namespace gpucast {
  namespace math {
    namespace domain {

template <typename value_t>
class contour_map_base
{
  public : // typedef / enums

    typedef value_t                                                 value_type;
    typedef point<value_type,2>                                     point_type;
    typedef axis_aligned_boundingbox<point_type>                    bbox_type;
    typedef interval<value_type>                                    interval_type;
    
    typedef contour<value_type>                                     contour_type;
    typedef std::shared_ptr<contour_type>                           contour_ptr;

    typedef std::vector<contour_ptr>                                contour_container;
    typedef contour_segment<value_type>                             contour_segment_type;
    typedef std::shared_ptr<contour_segment_type>                   contour_segment_ptr;

    typedef std::vector<contour_segment_ptr>                        contour_segment_container;
    typedef std::map<contour_ptr, contour_segment_container>        contour_segment_map;

    typedef beziercurve<point_type>                                 curve_type;
    typedef std::shared_ptr<curve_type>                             curve_ptr;

  public : // c'tor / d'tor

    contour_map_base           ();
    virtual ~contour_map_base  ();

  public : // methods

    void                              add                           ( contour_type const& loop );
    void                              clear                         ();
    contour_segment_map const&        loops() const;

    contour_segment_container const&  monotonic_segments   () const;

    std::size_t                       count_curves                  () const;
    std::vector<curve_ptr>            curves                        () const;
    bbox_type const&                  bounds                        () const;
    void                              update_bounds                 ();

  public : // virtual methods

    // stream output of domain
    virtual void                      print                         ( std::ostream& os ) const;

    // initial partitioning scheme according to given monotonic curves
    virtual void                      initialize                    () = 0;

  protected : // methods

    static void                       _determine_splits             ( std::set<value_type>&                       result, 
                                                                      typename point_type::coordinate_type const& dimension,
                                                                      contour_segment_container const&            contours );

    static void                       _intervals_from_splits        ( std::set<value_type> const& input,
                                                                      std::set<interval_type>&    output );

    std::size_t                       _contours_greater             ( value_type const& value,
                                                                      typename point_type::coordinate_type const& dimension,
                                                                      contour_segment_container const& input ) const;

    void                              _contours_in_interval         ( interval_type const& interval,
                                                                      typename point_type::coordinate_type const& dimension,
                                                                      contour_segment_container const&            input,
                                                                      contour_segment_container&                  output ) const;

  private : // internal/auxilliary methods

  protected : // attributes

    // geometrical data
    contour_segment_map               _segmented_loops;

    contour_segment_container         _contour_segments;

    // total bounds of domain
    bbox_type                         _bounds;
};


template <typename value_t>
std::ostream& operator<<(std::ostream& os, contour_map_base<value_t> const& rhs);

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#include "contour_map_base_impl.hpp"

#endif // GPUCAST_MATH_CONTOUR_MAP_BASE_HPP
