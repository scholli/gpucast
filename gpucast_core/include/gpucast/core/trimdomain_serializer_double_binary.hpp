/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_double_binary.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP
#define GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/double_binary/partition.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/conversion.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
struct GPUCAST_CORE trim_double_binary_serialization {

  trim_double_binary_serialization() :
    partition(1),
    celldata(1),
    curvelist(1),
    curvedata(1),
    preclassification(1)
  {}

  std::size_t size_in_bytes() const {
    return partition.size() * sizeof(math::vec4f) +
      celldata.size() * sizeof(math::vec4f) +
      curvelist.size() * sizeof(math::vec4f) +
      curvedata.size() * sizeof(math::vec3f) +
      preclassification.size() * sizeof(unsigned char);
  }

  std::vector<gpucast::math::vec4f> partition;         // "trimdata"    
  std::vector<gpucast::math::vec4f> celldata;          // "urangeslist" 
  std::vector<gpucast::math::vec4f> curvelist;         // "curvelist"   
  std::vector<gpucast::math::vec3f> curvedata;         // "curvedata"   
  std::vector<unsigned char>        preclassification; // "preclassification"

  std::unordered_map<trimdomain_ptr, trimdomain_serializer::address_type>         domain_index_map;
  std::unordered_map<trimdomain::curve_ptr, trimdomain_serializer::address_type>  curve_index_map;
};

class GPUCAST_CORE trimdomain_serializer_double_binary : public trimdomain_serializer
{
  public : // enums/typedefs

  public : // methods

    address_type     serialize  ( trimdomain_ptr const&              input, 
                                  trim_double_binary_serialization&  serialization,
                                  bool                               texture_classification_enabled = false,
                                  unsigned                           texture_classification_resolution = 8) const;

    // output_vslabs
    // [ id ]       [ id + 1 ]      ...   [ id + 2 + #vintervals ]
    // #vintervals   umin_total      vmin_0                     vmin_n
    // 0             umin_total      vmax_0                     vmax_n
    // 0             vmin_total      uid_0                      uid_n
    // 0             vmax_total      #uintervals_0              #uintervals_n

    // output_cells
    // [uid]             [ uid + 1 ]      ...   [ uid + 1 + #uintervals ]
    //  umin_total        umin_0                     umin_n
    //  umax_total        umax_0                     umax_n
    //  #uintervals_n     intersects_0               intersects_n
    //  0                 curvelist_id_0             curvelist_id_n

    // output_curvelists
    // [ curvelist_id ]    
    //    # curves         curve_id      curve_p0_u
    //    0                +/-curve_order   +/-curve_p0_v    => + means du/dt > 0; - means du/dt < 0
    //    0                tmin          curve_pn_u
    //    0                tmax          curve_pn_v

    // output_curvedata
    // [ curve_id ]  ... [ curve_id + curve_order ]
    //    wp0x                       wpnx
    //    wp0y                       wpny
    //    w0                         wn



    // "urangeslist"
    // "curvelist"  
    // "curvedata"  

  private : // member

};



} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP
