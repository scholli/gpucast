/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_loop_contour_list.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_LOOP_CONTOUR_LIST_HPP
#define GPUCAST_CORE_TRIMDOMAIN_LOOP_CONTOUR_LIST_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_loop_list.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/conversion.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

struct GPUCAST_CORE trim_loop_list_serialization
{
  trim_loop_list_serialization()
    : loops(1),
    contours(1),
    curves(1),
    points(1),
    preclassification(1)
  {}

  std::size_t size_in_bytes() const {
    return loops.size() * sizeof(loop_t) +
      contours.size() * sizeof(contour_t) +
      curves.size() * sizeof(curve_t) +
      points.size() * sizeof(point_t) +
      preclassification.size() * sizeof(unsigned char);
  }

  struct bbox_t {
    float umin;
    float umax;
    float vmin;
    float vmax;
  };

  struct point_t {
    float wx;
    float wy;
    float w;
    float pad;
  };

  struct curve_t {
    unsigned order;
    unsigned point_index;
    unsigned uincreasing;
    unsigned pad; // 16 byte alignment
    bbox_t   bbox;
  };

  struct loop_t {
    unsigned nchildren;
    unsigned child_index;
    unsigned ncontours;
    unsigned contour_index;

    unsigned pre_class_id;
    unsigned pre_class_width;
    unsigned pre_class_height;
    unsigned pad0; // 16 byte alignment

    float    umin;
    float    umax;
    float    vmin;
    float    vmax;

    bbox_t   bbox;
  };

  struct contour_t {
    unsigned ncurves;
    unsigned curve_index;
    unsigned uincreasing;
    unsigned parity_priority;
    bbox_t   bbox;
  };

  std::vector<loop_t>        loops;
  std::vector<contour_t>     contours;
  std::vector<curve_t>       curves;
  std::vector<point_t>       points;
  std::vector<unsigned char> preclassification;

  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type> domain_index_map;
};


class GPUCAST_CORE trimdomain_serializer_loop_contour_list : public trimdomain_serializer
{
  public : // enums/typedefs

    using trimdomain_serializer::address_type;

    typedef trimdomain::value_type                                   value_type;
    typedef gpucast::math::domain::contour_map_loop_list<value_type> partition_type;
    typedef partition_type::contour_segment_ptr                      contour_segment_ptr;
    typedef partition_type::contour_ptr                              contour_ptr;

  public : // methods

    address_type     serialize(trimdomain_ptr const& input,
                               trim_loop_list_serialization& result,
                               bool pre_classification_enabled,
                               unsigned pre_classification_resolution ) const;

    void             serialize(contour_segment_ptr const& input, partition_type const& partition, trim_loop_list_serialization& result) const;

    void             serialize(curve_ptr const& curve, trim_loop_list_serialization& result) const;

    address_type     serialize(partition_type::trimloop const& loop, partition_type const& partition, trim_loop_list_serialization& result, bool is_outer_loop) const;

  private : // member

};

std::ostream& operator<<(std::ostream& os, trim_loop_list_serialization::point_t const& p);
std::ostream& operator<<(std::ostream& os, trim_loop_list_serialization::bbox_t const& p);
std::ostream& operator<<(std::ostream& os, trim_loop_list_serialization::curve_t const& p);
std::ostream& operator<<(std::ostream& os, trim_loop_list_serialization::loop_t const& p);
std::ostream& operator<<(std::ostream& os, trim_loop_list_serialization::contour_t const& p);

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_LOOP_CONTOUR_LIST_HPP
