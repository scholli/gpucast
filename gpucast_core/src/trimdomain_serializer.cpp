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
trimdomain_serializer::trimdomain_serializer()
{}


/////////////////////////////////////////////////////////////////////////////
/* virtual */ trimdomain_serializer::~trimdomain_serializer()
{}


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

} // namespace gpucast
