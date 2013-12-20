/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testpoint.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#if WIN32
  #include <UnitTest++.h>
#else
  #include <unittest++/UnitTest++.h>
#endif

#include <vector>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/glsl.hpp>

using namespace gpucast::math;

SUITE (point_class)
{
  TEST(default_ctor)
  {
    point<double, 6> p;

    CHECK_EQUAL(p[0], 0.0);
    CHECK_EQUAL(p[1], 0.0);
    CHECK_EQUAL(p[2], 0.0);
    CHECK_EQUAL(p[3], 0.0);
    CHECK_EQUAL(p[4], 0.0);
    CHECK_EQUAL(p[5], 0.0);
    CHECK_EQUAL(p.weight(), 1.0);
  }

  TEST(ctor_wo_weight)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    point<double, 6> p1(data);

    for (unsigned i = 0; i != point<double, 6>::coordinates; ++i) 
    {
      CHECK_EQUAL(p0[i], 4.0);
      CHECK_EQUAL(p1[i], 4.0);
    }
    
    CHECK_EQUAL(p0.weight(), 1.0);
    CHECK_EQUAL(p1.weight(), 1.0);
  }

  TEST(copy_ctor)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);

    point<double, 6> p1 = p0;

    CHECK_EQUAL(p1.weight(), 2.0);
    CHECK_EQUAL(p0, p1);
  }

  TEST(assignment)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);

    point<double, 6> p1;
    p1 = p0;

    CHECK_EQUAL(p1.weight(), 2.0);
    CHECK_EQUAL(p0, p1);
  }

  TEST(addition)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);

    point<double, 6> p1 = p0;

    p1 += p0;

    for (unsigned i = 0; i != point<double, 6>::coordinates; ++i) 
    {
      CHECK_EQUAL(p1[i], 8.0);
    }

    CHECK_EQUAL(p1.weight(), 2.0);
  }

  TEST(subtraction)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);

    point<double, 6> p1 = p0;

    p1 -= p0;

    for (unsigned i = 0; i != point<double, 6>::coordinates; ++i) 
    {
      CHECK_EQUAL(p1[i], 0.0);
      CHECK_EQUAL(p0[i], 4.0);
    }

    CHECK_EQUAL(p1.weight(), 0.0);
    CHECK_EQUAL(p0.weight(), 2.0);
  }

  TEST(comparison)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);

    point<double, 6> p1 = p0;

    CHECK_EQUAL(p1, p0);

    p1.weight(3.0);

    CHECK( p1 != p0 );
  }

  TEST(swap)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);

    point<double, 6> p1 = p0;

    p1 -= p0;

    p1.swap(p0);

    for (unsigned i = 0; i != point<double, 6>::coordinates; ++i) 
    {
      CHECK_EQUAL(p0[i], 0.0);
      CHECK_EQUAL(p1[i], 4.0);
    }

    CHECK_EQUAL(p0.weight(), 0.0);
    CHECK_EQUAL(p1.weight(), 2.0);
  }

  TEST(multiplication_w_scalar)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);
    p0 *= 3.0;

    for (unsigned i = 0; i != point<double, 6>::coordinates; ++i) 
    {
      CHECK_EQUAL(p0[i], 12.0);
    }

    CHECK_EQUAL(p0.weight(), 6.0);
  }

  TEST(abs)
  {
    std::vector<double> data(6, 4.0);
    point<double, 6> p0(data);
    p0.weight(2.0);

    CHECK_CLOSE(p0.abs(), sqrt(16.0*6.0), 0.0001);

    point<float, 5> p1;
    p1[0] = 0.3f;
    p1[1] = -2.6f;
    p1[2] = 1.2f;
    p1[3] = 5.6f;
    p1[4] = 0.0f;

    CHECK_CLOSE(std::sqrt( 0.3f *  0.3f + 
                          (-2.6f) * (-2.6f) +
                           1.2f *  1.2f +
                           5.6f *  5.6f + 
                           0.0f *  0.0f ), p1.abs(), 0.00001);
  }


  TEST(normalize)
  {
    point<float, 5> p1;
    p1[0] = 0.3f;
    p1[1] = -2.6f;
    p1[2] = 1.2f;
    p1[3] = 5.6f;
    p1[4] = 0.0f;

    p1.normalize();

    CHECK_CLOSE(p1.abs(), 1.0f, 0.000001);
  }
}
