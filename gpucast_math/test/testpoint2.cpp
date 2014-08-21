/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testpoint2.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <cmath>

#include <gpucast/math/parametric/point.hpp>

using namespace gpucast::math;

SUITE (point2_class)
{
  TEST(default_ctor)
  {
    point2d p;

    CHECK_EQUAL(p[0], 0.0);
    CHECK_EQUAL(p[1], 0.0);
    CHECK_EQUAL(p.weight(), 1.0);
  }

  TEST(ctor_wo_weight)
  {
    point2d p(2, 5);

    CHECK_EQUAL(p[0], 2.0);
    CHECK_EQUAL(p[1], 5.0);
    CHECK_EQUAL(p.weight(), 1.0);
  }

  TEST(ctor_w_weight)
  {
    point2d p(4, 5, 3.0);

    CHECK_EQUAL(p[0], 4.0);
    CHECK_EQUAL(p[1], 5.0);
    CHECK_EQUAL(p.weight(), 3.0);
  }

  TEST(assignment)
  {
    point2d p(4, 5, 3.0);
    point2d p2;
    p2 = p;

    CHECK_EQUAL(p[0], 4.0);
    CHECK_EQUAL(p[1], 5.0);
    CHECK_EQUAL(p.weight(), 3.0);

    CHECK_EQUAL(p[0], p2[0]);
    CHECK_EQUAL(p[1], p2[1]);
    CHECK_EQUAL(p.weight(), p2.weight());
  }

  TEST(addition)
  {
    point2d p(4, 5, 3.0);
    point2d p2(2, 2);
    p += p2;

    CHECK_EQUAL(p[0], 6);
    CHECK_EQUAL(p[1], 7);
    CHECK_EQUAL(p.weight(), 4);
  }

  TEST(subtraction)
  {
    point2d p(4, 5, 3.0);
    point2d p2(2, 2);
    p -= p2;

    CHECK_EQUAL(p[0], 2);
    CHECK_EQUAL(p[1], 3);
    CHECK_EQUAL(p.weight(), 2);
  }

  TEST(comparison)
  {
    point2d p1(4, 5, 3.0);
    point2d p2(2, 2);
    point2d p3(2, 2, 2.0);
    point2d p4(0, 0, 1);
    point2d p5;
    point2d p6(2, 2, 1.0);

    CHECK( p1 == p1);
    CHECK( p2 != p1);
    CHECK( p3 != p1);
    CHECK( p4 != p1);
    CHECK( p5 != p1);
    CHECK( p6 != p1);

    CHECK( p1 != p2);
    CHECK( p2 == p2);
    CHECK( p3 != p2);
    CHECK( p4 != p2);
    CHECK( p5 != p2);
    CHECK( p6 == p2);

    CHECK( p1 != p3);
    CHECK( p2 != p3);
    CHECK( p3 == p3);
    CHECK( p4 != p3);
    CHECK( p5 != p3);
    CHECK( p6 != p3);

    CHECK( p1 != p4);
    CHECK( p2 != p4);
    CHECK( p3 != p4);
    CHECK( p4 == p4);
    CHECK( p5 == p4);
    CHECK( p6 != p4);

    CHECK( p1 != p5);
    CHECK( p2 != p5);
    CHECK( p3 != p5);
    CHECK( p4 == p5);
    CHECK( p5 == p5);
    CHECK( p6 != p5);

    CHECK( p1 != p6);
    CHECK( p2 == p6);
    CHECK( p3 != p6);
    CHECK( p4 != p6);
    CHECK( p5 != p6);
    CHECK( p6 == p6 );
  }

  TEST(swap)
  {
    point2d p(4, 5, 3.0);
    point2d p2(2, 2);

    p.swap(p2);

    CHECK_EQUAL(p[0], 2);
    CHECK_EQUAL(p[1], 2);
    CHECK_EQUAL(p.weight(), 1);

    CHECK_EQUAL(p2[0], 4);
    CHECK_EQUAL(p2[1], 5);
    CHECK_EQUAL(p2.weight(), 3);
  }

  TEST(multiplication_w_scalar)
  {
    point2d p(4, 5, 3.0);
    p *= 2.0;

    point2d p2 = p * 2.0;

    point2d p3 = 2.0 * p2;

    CHECK_EQUAL(p[0], 8);
    CHECK_EQUAL(p[1], 10);
    CHECK_EQUAL(p.weight(), 6);

    CHECK_EQUAL(p2[0], 16);
    CHECK_EQUAL(p2[1], 20);
    CHECK_EQUAL(p2.weight(), 12);

    CHECK_EQUAL(p3[0], 32);
    CHECK_EQUAL(p3[1], 40);
    CHECK_EQUAL(p3.weight(), 24);
  }

  TEST(abs)
  {
    point2d p1(4, 5, 3.0);
    point2d p2(1, 4, 3.0);
    point2d p3(-3, 2, 3.0);

    CHECK_CLOSE(p1.abs(), std::sqrt(41.0f), 0.0001);
    CHECK_CLOSE(p2.abs(), std::sqrt(17.0f), 0.0001);
    CHECK_CLOSE(p3.abs(), std::sqrt(13.0), 0.0001);
  }

}
