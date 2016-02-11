/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_kd.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_CONTOUR_MAP_KD_HPP
#define GPUCAST_MATH_CONTOUR_MAP_KD_HPP

// includes, system

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_base.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdtree2d.hpp>
#include <gpucast/math/parametric/domain/partition/classification_field.hpp>

namespace gpucast {

  enum kd_split_strategy {
    midsplit,
    greedy,
    maxarea,
    sah,
    minsplits
  };

  namespace math {
    namespace domain {

template <typename value_t>
class contour_map_kd : public contour_map_base<value_t>
{
  public : // typedef / enums

    typedef contour_map_base<value_t>              base_type;
    using typename contour_map_base<value_t>::interval_type;
    using typename contour_map_base<value_t>::contour_segment_container;
    using typename contour_map_base<value_t>::value_type;
    using typename contour_map_base<value_t>::point_type;
    using typename contour_map_base<value_t>::bbox_type;
    using typename contour_map_base<value_t>::contour_segment_ptr;
    using typename contour_map_base<value_t>::contour_type;
    using typename contour_map_base<value_t>::contour_segment_type;
    typedef kdtree2d<value_t> kdtree_type;
    typedef kdnode2d<value_t> kdnode_type;
    typedef typename kdnode_type::kdnode_ptr kdnode_ptr;

  public : // helper types

  public : // c'tor / d'tor

    contour_map_kd(kd_split_strategy s, bool usebitfield = false, unsigned bitfield_resolution = 8);

  public : // methods

    /*virtual*/ bool initialize() override;

    kdtree2d<value_t> const& partition() const;

    classification_field<unsigned char> const& pre_classification() const;

    /*virtual*/ void print(std::ostream& os) const;

  protected : // methods

  private : // attributes

    kdtree2d<value_t>                   _kdtree;
    kd_split_strategy                   _split_strategy;

    bool                                _enable_bitfield;
    classification_field<unsigned char> _classification_texture;
};

template <typename value_t>
std::ostream& operator<<(std::ostream& os, contour_map_kd<value_t> const& rhs);

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#include "contour_map_kd_impl.hpp"

#endif // GPUCAST_MATH_CONTOUR_MAP_KD_HPP
