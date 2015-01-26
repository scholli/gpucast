/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_HPP
#define GPUCAST_CORE_TRIMDOMAIN_HPP

#ifdef _MSC_VER
 #pragma warning(disable: 4251)
#endif

// header, system
#include <vector>
#include <iostream>

// header, external
#include <gpucast/math/vec4.hpp>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_segment.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>


namespace gpucast {

class GPUCAST_CORE trimdomain
{
  public : // enums/typedefs

    typedef double                                     value_type;                               
    typedef gpucast::math::point<double,2>             point_type;

    typedef gpucast::math::domain::contour_segment<value_type> contour_segment_type;
    typedef std::shared_ptr<contour_segment_type>      contour_segment_ptr;
    typedef std::shared_ptr<trimdomain>                domain_ptr;

    typedef gpucast::math::beziercurve<point_type>     curve_type;
    typedef std::shared_ptr<curve_type>                curve_ptr;
    typedef std::vector<curve_ptr>                     curve_container;

    typedef gpucast::math::axis_aligned_boundingbox<point_type> bbox_type;
    typedef gpucast::math::domain::contour<value_type> contour_type;

    typedef std::vector<contour_type>                  trimloop_container;

  public : // c'tor/d'tor

    trimdomain();
    ~trimdomain();

    void swap(trimdomain& swp);

  public : // methods

    // add loop
    void                      add           ( contour_type const& loop );

    std::size_t               size          () const;

    bool                      empty         () const;

    void                      nurbsdomain   ( bbox_type const& );
    bbox_type const&          nurbsdomain   () const;

    // true = trim outer values; false = trim inner values
    bool                      type          () const;
    void                      type          ( bool inner );

    curve_container           curves        () const;
    std::size_t               loop_count    () const;

    trimloop_container const& loops         () const;

    void                      print         ( std::ostream& os ) const;

  private : // member

    // curves and type(inner outer trim)
    trimloop_container   _trimloops;
    bool                 _type;
    bbox_type            _nurbsdomain;  // [umin, umax, vmin, vmax]
  };

GPUCAST_CORE std::ostream&
operator<<(std::ostream& os, trimdomain const& t);

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_HPP
