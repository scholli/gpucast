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

template <typename T>
class grid : public std::vector<T> {
public :
  grid(unsigned w, unsigned h) : std::vector<T>(w*h, T()), _width(w), _height(h) {}

  T& operator()(unsigned x, unsigned y) {
    return (*this)[x + y*_width];
  }
  T const& operator()(unsigned x, unsigned y) const {
    return (*this)[x + y*_width];
  }

  unsigned width() const { return _width; }
  unsigned height() const { return _height; }

private: 
  unsigned _width;
  unsigned _height;
};


class GPUCAST_CORE trimdomain
{
  public : // enums/typedefs

    enum pre_classification {
      unknown   = 0,
      untrimmed = 1,
      trimmed   = 2
    };

    typedef double                                     value_type;                               
    typedef gpucast::math::point<value_type, 2>        point_type;

    typedef gpucast::math::domain::contour_segment<value_type> contour_segment_type;
    typedef std::shared_ptr<contour_segment_type>      contour_segment_ptr;
    typedef std::shared_ptr<trimdomain>                domain_ptr;

    typedef gpucast::math::beziercurve<point_type>     curve_type;
    typedef curve_type::bbox_type                      bbox_type;

    typedef std::shared_ptr<curve_type>                curve_ptr;
    typedef std::vector<curve_ptr>                     curve_container;

    typedef gpucast::math::domain::contour<value_type> contour_type;

    typedef std::vector<contour_type>                  trimloop_container;

  public : // c'tor/d'tor

    trimdomain();

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

    void                      normalize     ();

    curve_container           curves        () const;
    std::size_t               loop_count    () const;
    std::size_t               max_degree    () const;

    trimloop_container const& loops         () const;
    
    grid<value_type>          signed_distance_field(unsigned resolution) const;
    value_type                signed_distance(point_type const& point) const;

    grid<unsigned char>       signed_distance_pre_classification(unsigned resolution) const;
    grid<unsigned char>       pre_classification(unsigned resolution) const;

    void                      print         ( std::ostream& os ) const;

  private : // member

    // curves and type(inner outer trim)
    mutable std::map<unsigned, grid<value_type>> _signed_distance_fields;
    trimloop_container                          _trimloops;
    bool                                        _type;
    bbox_type                                   _nurbsdomain;  // [umin, umax, vmin, vmax]
  };

typedef std::shared_ptr<trimdomain> trimdomain_ptr;

GPUCAST_CORE std::ostream&
operator<<(std::ostream& os, trimdomain const& t);

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_HPP
