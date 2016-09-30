#ifndef GPUCAST_GLSL_CONVERSION
#define GPUCAST_GLSL_CONVERSION

/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : conversion.frag
*  project    : gpucast
*  description:
*
********************************************************************************/

uvec4 intToUInt4 ( uint input )
{
  uvec4 result;
  result.w = (input & 0xFF000000) >> 24U;
  result.z = (input & 0x00FF0000) >> 16U;
  result.y = (input & 0x0000FF00) >> 8U;
  result.x = (input & 0x000000FF);
  return result;
}

uint uint4ToUInt ( in uvec4 input )
{
  uint result = 0U;
  result |= (input.w & 0x000000FF) << 24U;
  result |= (input.z & 0x000000FF) << 16U;
  result |= (input.y & 0x000000FF) << 8U;
  result |= (input.x & 0x000000FF);
  return result;
}

uint uint2ToUInt ( in uvec2 input )
{
  uint result = 0U;
  result |= (input.y & 0x0000FFFF) << 16U;
  result |= (input.x & 0x0000FFFF);
  return result;
}

int uint2ToInt ( in uvec2 input )
{
  int result = 0;
  result |= (int(input.y) & 0x0000FFFF) << 16U;
  result |= (int(input.x) & 0x0000FFFF);
  return result;
}

uint bvec4ToUInt ( in bvec4 input )
{
  uint result = 0U;
  result |= (uint(input.w) & 0x00000001) << 3U;
  result |= (uint(input.z) & 0x00000001) << 2U;
  result |= (uint(input.y) & 0x00000001) << 1U;
  result |= (uint(input.x) & 0x00000001);
  return result;
}

bvec4 uintToBvec4 ( in uint input ) 
{
  bvec4 result;
  result.w = bool((input & 0x00000008) >> 3U);
  result.z = bool((input & 0x00000004) >> 2U);
  result.y = bool((input & 0x00000002) >> 1U);
  result.x = bool((input & 0x00000001)      );
  return result;
}

uvec2 intToUInt2(in uint input)
{
  uvec2 result;
  result.y = (input & 0xFFFF0000) >> 16U;
  result.x = (input & 0x0000FFFF);
  return result;
}

void intToUint8_24 ( in  uint input,
                     out uint a,
                     out uint b )
{
  b = (input & 0xFFFFFF00) >> 8U;
  a = (input & 0x000000FF);
}

#endif