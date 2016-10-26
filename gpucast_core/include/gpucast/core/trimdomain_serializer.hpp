/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_HPP
#define GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_HPP

// header, system
#include <unordered_map>

#include <gpucast/math/halffloat.hpp>
#include <gpucast/math/vec4.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/util.hpp>
#include <gpucast/core/beziersurface.hpp>
#include <gpucast/core/hyperspace_adapter.hpp>

namespace gpucast {

class GPUCAST_CORE trimdomain_serializer
{
  public : // enums/typedefs

    typedef unsigned                                            address_type;
    typedef float                                               float_type;

    typedef beziersurface::curve_point_type                     point_type;
    typedef gpucast::math::axis_aligned_boundingbox<point_type> bbox_type;
    typedef beziersurface::curve_type                           curve_type;
    typedef std::shared_ptr<curve_type>                         curve_ptr;

    typedef beziersurface::trimdomain_ptr                       trimdomain_ptr;

  public : // c'tor/d'tor

  public : // methods

    address_type    serialize ( curve_ptr const&                             input_curve, 
                                std::unordered_map<curve_ptr, address_type>& referenced_curves,
                                std::vector<math::vec3f>&                    output_container ) const;

    address_type    serialize(trimdomain_ptr const& input_domain, 
                              std::vector<unsigned char>& output_classification_field, 
                              unsigned texture_classification_resolution) const;
    
    static float_type      unsigned_bits_as_float  ( address_type i );

    static address_type    float_bits_as_unsigned  ( float_type f );

    static address_type    uint4ToUInt (unsigned char a, unsigned char b, unsigned char c, unsigned char d);
    static address_type    uint8_24ToUInt (unsigned char a, unsigned int b);
    static void            intToUint8_24 (address_type input, unsigned char& a, unsigned int& b);
    static address_type    float2_to_unsigned (float a, float b);

  private : // member



  };

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_HPP
