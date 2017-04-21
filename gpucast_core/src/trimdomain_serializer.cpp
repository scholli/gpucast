/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/trimdomain_serializer.hpp"

// header, system

// header, project
#include <gpucast/core/hyperspace_adapter.hpp>

namespace gpucast {


/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type trimdomain_serializer::serialize(curve_ptr const& input_curve,
                                                                     std::unordered_map<curve_ptr, address_type>& referenced_curves,
                                                                     std::vector<math::vec3f>&                      output_container) const
{
  // find curve index, if already referenced
  std::unordered_map<curve_ptr, address_type>::const_iterator curve_index = referenced_curves.find(input_curve);

  if (curve_index != referenced_curves.end())
  {
    return curve_index->second;
  }
  else
  {
    // save current index
    address_type index = address_type(output_container.size());

    // copy curve data into buffer
    std::transform(input_curve->begin(),
      input_curve->end(),
      std::back_inserter(output_container),
      hyperspace_adapter_2D_to_3D<gpucast::math::point2d, math::vec3f>());

    // insert curve pointer and according index into map
    referenced_curves.insert(std::make_pair(input_curve, index));

    if (output_container.size() >= std::numeric_limits<address_type>::max())
    {
      throw std::runtime_error("Address exceeds maximum of addressable memory");
    }

    // return index the curve was written to
    return index;
  }
}


/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type  trimdomain_serializer::serialize(trimdomain_ptr const& input_domain,
  std::vector<unsigned char>& output_classification_field,
  unsigned texture_classification_resolution) const
{
  address_type classification_id = address_type(output_classification_field.size());

#if 0
  // get pre-classified texture data
  auto pre_classification = input_domain->signed_distance_pre_classification(texture_classification_resolution);
#else
  auto pre_classification = input_domain->pre_classification(texture_classification_resolution);
#endif
  // copy to global field
  std::copy(pre_classification.begin(), pre_classification.end(), std::back_inserter(output_classification_field));
  
  return classification_id;
}


/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::float_type     
trimdomain_serializer::unsigned_bits_as_float ( address_type i )
{
  assert ( sizeof ( float_type ) == sizeof ( address_type ) );

  float_type as_float = *reinterpret_cast<float_type*>(&i);
  return as_float;
}


/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type  
trimdomain_serializer::float_bits_as_unsigned ( float_type f )
{
  assert ( sizeof ( float_type ) == sizeof ( address_type ) );

  address_type as_unsigned = *reinterpret_cast<address_type*>(&f);
  return as_unsigned;
}

/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type 
trimdomain_serializer::uint4ToUInt(unsigned char a, unsigned char b, unsigned char c, unsigned char d)
{
  assert(sizeof(address_type) == 4);

  address_type result = 0U;
  result |= (d & 0x000000FF) << 24U;
  result |= (c & 0x000000FF) << 16U;
  result |= (b & 0x000000FF) << 8U;
  result |= (a & 0x000000FF);

  return result;
}

/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type 
trimdomain_serializer::uint8_24ToUInt(unsigned char a, unsigned int b)
{
  assert(sizeof(address_type) == 4);

  address_type result = 0U;
  result |= (b & 0x00FFFFFF) << 8U;
  result |= (a & 0x000000FF);

  return result;
}

/////////////////////////////////////////////////////////////////////////////
void 
trimdomain_serializer::intToUint8_24(trimdomain_serializer::address_type input,
  unsigned char& a,
  unsigned int& b)
{
  b = (input & 0xFFFFFF00) >> 8U;
  a = (input & 0x000000FF);
}

/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type 
trimdomain_serializer::float2_to_unsigned(float a, float b)
{
  gpucast::math::halffloat_t ah = gpucast::math::floatToHalf(a);
  gpucast::math::halffloat_t bh = gpucast::math::floatToHalf(b);

  address_type result = 0U;
  result |= (bh & 0x0000FFFF) << 16U;
  result |= (ah & 0x0000FFFF);

  return result;
}

} // namespace gpucast
