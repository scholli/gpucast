/********************************************************************************
*
* Copyright (C) 2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testgcd.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#if WIN32
  #include <UnitTest++.h>
#else
  #include <unittest++/UnitTest++.h>
#endif

#include <iostream>
#include <gpucast/math/gcd.hpp>

using namespace gpucast::math;



SUITE (gcd)
{

  TEST(meta_type)
  {
    CHECK_EQUAL(long(gpucast::math::gcd_t<2002,  210>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<2002, -210>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<-2002,-210>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<-2002, 210>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<210,  2002>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<210, -2002>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<-210, 2002>::result), 14);
    CHECK_EQUAL(long(gpucast::math::gcd_t<-210,-2002>::result), 14);
  }

  TEST(function)
  {
    CHECK(gcd(-2002, 210) == 14);
    CHECK(gcd(-2002, -210) == 14);
    CHECK(gcd(2002, -210) == 14);
    CHECK(gcd(2002, 210) == 14);
    CHECK(gcd(-210, -2002) == 14);
    CHECK(gcd(-210, 2002) == 14);
    CHECK(gcd(210, -2002) == 14);
    CHECK(gcd(210, 2002) == 14);
  }

}
