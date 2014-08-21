/********************************************************************************
*
* Copyright (C) 2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testpow.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <gpucast/math/pow.hpp>

using namespace gpucast::math;

#define MAX_UINT 4294967295 	


SUITE (pow)
{

  TEST(metatype)
  {
    CHECK(int(gpucast::math::meta_pow<2,8>::result) == 256);
  }


}
