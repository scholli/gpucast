/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : conversion.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_CONVERSION_HPP
#define GPUCAST_CORE_CONVERSION_HPP

// header, system
#include <array>

// header, project

namespace gpucast {

  //////////////////////////////////////////////////////////////////////////////
  template <typename source_t, typename target_t>
  target_t explicit_type_conversion(source_t const& t) {
    return target_t(t);
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename source_t, typename target_t>
  target_t const bit_cast(source_t const& s)
  {
    assert(sizeof (source_t) == sizeof(target_t));

    return *reinterpret_cast<target_t const*>(&s);
  }

  //////////////////////////////////////////////////////////////////////////////
  inline unsigned
  uint4x8ToUInt(unsigned char input0, unsigned char input1, unsigned char input2, unsigned char input3)
  {
    unsigned result = 0U;
    result |= (input3 & 0x000000FF) << 24U;
    result |= (input2 & 0x000000FF) << 16U;
    result |= (input1 & 0x000000FF) << 8U;
    result |= (input0 & 0x000000FF);
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////
  inline std::array<unsigned short, 4>
  uintToUInt4x(unsigned int input)
{
    std::array<unsigned short, 4> result;
    result[3] = (input & 0xFF000000) >> 24U;
    result[2] = (input & 0x00FF0000) >> 16U;
    result[1] = (input & 0x0000FF00) >> 8U;
    result[0] = (input & 0x000000FF);
    return result;
  }


  //////////////////////////////////////////////////////////////////////////////
  inline unsigned
  uint2x16ToUInt(unsigned short input0, unsigned short input1)
  {
    unsigned result = 0U;
    result |= (input1 & 0x0000FFFF) << 16U;
    result |= (input0 & 0x0000FFFF);
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////
  inline
  std::array<unsigned short, 2> uintToUint2x16(unsigned input)
  {
    std::array<unsigned short, 2> result;
    result[1] = (input & 0xFFFF0000) >> 16U;
    result[0] = (input & 0x0000FFFF);
    return result;
  }

}

#endif // GPUCAST_CORE_CONVERSION_HPP
