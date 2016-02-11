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
trimdomain_serializer::address_type  trimdomain_serializer::serialize(trimdomain_ptr const& input_domain,
  std::vector<unsigned char>& output_classification_field,
  unsigned texture_classification_resolution) const
{
  address_type classification_id = address_type(output_classification_field.size());

  auto signed_distance_field = input_domain->signed_distance_field(texture_classification_resolution);
  auto texel_diameter = input_domain->nurbsdomain().size() / texture_classification_resolution;
  auto texel_radius = texel_diameter.abs() / 2;

  std::transform(signed_distance_field.begin(), signed_distance_field.end(), std::back_inserter(output_classification_field), [texel_radius](double v){
    if (v < -texel_radius)
      return trimdomain::trimmed; // classified outside
    if (v > texel_radius)
      return trimdomain::untrimmed; // classified inside
    else
      return trimdomain::unknown; // not classified
  });
  
  return classification_id;
}


/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::float_type     
trimdomain_serializer::unsigned_bits_as_float ( address_type i ) const
{
  assert ( sizeof ( float_type ) == sizeof ( address_type ) );

  float_type as_float = *reinterpret_cast<float_type*>(&i);
  return as_float;
}


/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type  
trimdomain_serializer::float_bits_as_unsigned ( float_type f ) const
{
  assert ( sizeof ( float_type ) == sizeof ( address_type ) );

  address_type as_unsigned = *reinterpret_cast<address_type*>(&f);
  return as_unsigned;
}

/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type 
trimdomain_serializer::uint4ToUInt(unsigned char a, unsigned char b, unsigned char c, unsigned char d) const
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
trimdomain_serializer::uint8_24ToUInt(unsigned char a, unsigned int b) const
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
  unsigned int& b) const
{
  b = (input & 0xFFFFFF00) >> 8U;
  a = (input & 0x000000FF);
}

/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type 
trimdomain_serializer::float2_to_unsigned(float a, float b) const
{
  gpucast::math::halffloat_t ah = gpucast::math::floatToHalf(a);
  gpucast::math::halffloat_t bh = gpucast::math::floatToHalf(b);

  address_type result = 0U;
  result |= (bh & 0x0000FFFF) << 16U;
  result |= (ah & 0x0000FFFF);

  return result;
}

} // namespace gpucast
