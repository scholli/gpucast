/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_contour_map_binary.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_CONTOUR_MAP_BINARY_HPP
#define GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_CONTOUR_MAP_BINARY_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_binary.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
struct GPUCAST_CORE trim_contour_binary_serialization {

  trim_contour_binary_serialization() 
  : partition(1),
    contourlist(1),
    curvelist(1),
    curvedata(1),
    pointdata(1),
    preclassification(1)
  {}

  std::size_t size_in_bytes() const {
    return partition.size() * sizeof(math::vec4f) +
      contourlist.size() * sizeof(math::vec4f) +
      curvelist.size() * sizeof(math::vec4f) +
      curvedata.size() * sizeof(float) +
      pointdata.size() * sizeof(math::vec3f) +
      preclassification.size() * sizeof(unsigned char);
  }

  std::vector<gpucast::math::vec4f> partition;         // "uniform samplerBuffer sampler_partition;"
  std::vector<gpucast::math::vec4f> contourlist;       // "uniform samplerBuffer sampler_contourlist;"
  std::vector<gpucast::math::vec4f> curvelist;         // "uniform samplerBuffer sampler_curvelist;"
  std::vector<float>                curvedata;         // "uniform samplerBuffer sampler_curvedata;"
  std::vector<gpucast::math::vec3f> pointdata;         // "uniform samplerBuffer sampler_pointdata;"
  std::vector<unsigned char>        preclassification; // "uniform usamplerBuffer sampler_preclass;"

  std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type>                  domain_index_map;
  std::unordered_map<trimdomain::curve_ptr, trimdomain_serializer::address_type>           curve_index_map;
  std::unordered_map<trimdomain::contour_segment_ptr, trimdomain_serializer::address_type> contour_index_map;
};

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_CORE trimdomain_serializer_contour_map_binary : public trimdomain_serializer
{
  public : // enums/typedefs
 
    typedef trimdomain::value_type                                                      value_type;
    typedef gpucast::math::domain::contour_map_binary<value_type>::contour_segment_ptr  contour_segment_ptr;

  public : // methods

    address_type     serialize  ( trimdomain_ptr const&                                  input_domain, 
                                  trim_contour_binary_serialization&                     serialization,
                                  bool                                                   pre_classification_enabled = false,
                                  unsigned                                               pre_classification_resolution = 8) const;

    address_type     serialize_contour_segment  ( contour_segment_ptr const&         contour_segment,
                                                  trim_contour_binary_serialization& serialization ) const;
  private : // member

};



} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP
