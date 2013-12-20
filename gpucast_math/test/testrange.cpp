/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testinterval.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#if WIN32
  #include <UnitTest++.h>
#else
  #include <unittest++/UnitTest++.h>
#endif

#include <gpucast/math/interval.hpp>

using namespace gpucast::math;

SUITE (interval_class)
{
  TEST(ctor_dtor_copy)
  {
    interval<double> r(1.0, 3.0);

    CHECK(r.minimum() == 1.0);
    CHECK(r.maximum() == 3.0);
    CHECK(r.lower_boundary_type() == excluded);
    CHECK(r.upper_boundary_type() == excluded);

    interval<double> r2(-3.0, 2.0, included, included);

    CHECK(r2.minimum() == -3.0);
    CHECK(r2.maximum() ==  2.0);
    CHECK(r2.lower_boundary_type() == included);
    CHECK(r2.upper_boundary_type() == included);
  }

  TEST(operator_less)
  {
    interval<double> r0(2.0, 3.0, included, included);
    interval<double> r1(3.0, 4.0, excluded, included);
    interval<double> r2(2.0, 3.0, excluded, excluded);
    interval<double> r3(4.0, 5.0, included, excluded);
    interval<double> r4(1.0, 2.0, excluded, excluded);

    CHECK(!(r0 < r0));
    CHECK(!(r1 < r0));
    CHECK(!(r2 < r0));
    CHECK(!(r3 < r0));
    CHECK((r4 < r0));

    CHECK((r0 < r1));
    CHECK(!(r1 < r1));
    CHECK((r2 < r1));
    CHECK(!(r3 < r1));
    CHECK((r4 < r1));

    CHECK(!(r0 < r2));
    CHECK(!(r1 < r2));
    CHECK(!(r2 < r2));
    CHECK(!(r3 < r2));
    CHECK((r4 < r2));

    CHECK((r0 < r3));
    CHECK(!(r1 < r3));
    CHECK((r2 < r3));
    CHECK(!(r3 < r3));
    CHECK((r4 < r3));

    CHECK(!(r0 < r4));
    CHECK(!(r1 < r4));
    CHECK(!(r2 < r4));
    CHECK(!(r3 < r4));
    CHECK(!(r4 < r4));
  }

  TEST(operator_greater)
  {
    interval<double> r0(2.0, 3.0, included, included);
    interval<double> r1(3.0, 4.0, excluded, included);
    interval<double> r2(2.0, 3.0, excluded, excluded);
    interval<double> r3(4.0, 5.0, included, excluded);
    interval<double> r4(1.0, 2.0, excluded, excluded);

    CHECK(!(r0 > r0));
    CHECK((r1 > r0));
    CHECK(!(r2 > r0));
    CHECK((r3 > r0));
    CHECK(!(r4 > r0));

    CHECK(!(r0 > r1));
    CHECK(!(r1 > r1));
    CHECK(!(r2 > r1));
    CHECK(!(r3 > r1));
    CHECK(!(r4 > r1));

    CHECK(!(r0 > r2));
    CHECK((r1 > r2));
    CHECK(!(r2 > r2));
    CHECK((r3 > r2));
    CHECK(!(r4 > r2));

    CHECK(!(r0 > r3));
    CHECK(!(r1 > r3));
    CHECK(!(r2 > r3));
    CHECK(!(r3 > r3));
    CHECK(!(r4 > r3));

    CHECK((r0 > r4));
    CHECK((r1 > r4));
    CHECK((r2 > r4));
    CHECK((r3 > r4));
    CHECK(!(r4 > r4));
  }

  TEST(operator_equal_notequal)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(2.0, 3.0, excluded, included);
    interval<double> i2(2.0, 3.0, included, excluded);
    interval<double> i3(2.0, 3.0, included, included);

    CHECK(i0 == i0);
    CHECK(i0 != i1);
    CHECK(i0 != i2);
    CHECK(i0 != i3);

    CHECK(i1 != i0);
    CHECK(i1 == i1);
    CHECK(i1 != i2);
    CHECK(i1 != i3);

    CHECK(i2 != i0);
    CHECK(i2 != i1);
    CHECK(i2 == i2);
    CHECK(i2 != i3);

    CHECK(i3 != i0);
    CHECK(i3 != i1);
    CHECK(i3 != i2);
    CHECK(i3 == i3);
  }

  TEST(overlap)
  {
    interval<double> i0(1.0, 3.0, excluded, excluded);
    interval<double> i1(0.0, 0.1, excluded, included);
    interval<double> i2(0.0, 2.0, included, excluded);
    interval<double> i3(2.0, 5.0, included, included);
    interval<double> i4(4.0, 4.5, included, included);

    CHECK(i0.overlap(i0));
    CHECK(!i0.overlap(i1));
    CHECK(i0.overlap(i2));
    CHECK(i0.overlap(i3));
    CHECK(!i0.overlap(i4));

    CHECK(!i1.overlap(i0));
    CHECK(i1.overlap(i1));
    CHECK(i1.overlap(i2));
    CHECK(!i1.overlap(i3));
    CHECK(!i1.overlap(i4));

    CHECK(i2.overlap(i0));
    CHECK(i2.overlap(i1));
    CHECK(i2.overlap(i2));
    CHECK(!i2.overlap(i3));
    CHECK(!i2.overlap(i4));

    CHECK(i3.overlap(i0));
    CHECK(!i3.overlap(i1));
    CHECK(!i3.overlap(i2));
    CHECK(i3.overlap(i3));
    CHECK(i3.overlap(i4));

    CHECK(!i4.overlap(i0));
    CHECK(!i4.overlap(i1));
    CHECK(!i4.overlap(i2));
    CHECK(i4.overlap(i3));
    CHECK(i4.overlap(i4));
  }

  TEST(minimum)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(4.0, 1.0, excluded, included);

    CHECK_EQUAL(i0.minimum(), 2.0);
    CHECK_EQUAL(i1.minimum(), 1.0);
  }

  TEST(maximum)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(4.0, 1.0, excluded, included);

    CHECK_EQUAL(i0.maximum(), 3.0);
    CHECK_EQUAL(i1.maximum(), 4.0);
  }


  TEST(distance)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(1.0, 4.0, excluded, included);
    interval<double> i2(6.0, 8.0, excluded, included);

    CHECK_EQUAL(i0.distance(2.5), 0.0);
    CHECK_EQUAL(i0.distance(2.0), 0.0);
    CHECK_EQUAL(i0.distance(3.0), 0.0);

    CHECK_EQUAL(i0.distance(i0), 0.0);
    CHECK_EQUAL(i0.distance(i1), 0.0);
    CHECK_EQUAL(i0.distance(i2), 3.0);
    CHECK_EQUAL(i1.distance(i0), 0.0);
    CHECK_EQUAL(i1.distance(i1), 0.0);
    CHECK_EQUAL(i1.distance(i2), 2.0);
    CHECK_EQUAL(i2.distance(i0), 3.0);
    CHECK_EQUAL(i2.distance(i1), 2.0);
    CHECK_EQUAL(i2.distance(i2), 0.0);    
  }



  TEST(in_interval)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(4.0, 1.0, excluded, included);
    interval<double> i2(7.0, 8.0, included, included);

    CHECK(!i0.in(3.0));
    CHECK(!i0.in(2.0));
    CHECK(!i0.in(3.5));
    CHECK(i0.in(2.4));
    CHECK(!i0.in(1.5));

    CHECK(i1.in(1.0));
    CHECK(!i1.in(4.0));
    CHECK(!i1.in(0.3));
    CHECK(!i1.in(5.3));
    CHECK(i1.in(3.4));

    CHECK(i2.in(7.0));
    CHECK(i2.in(8.0));
    CHECK(i2.in(7.6));
    CHECK(!i2.in(8.1));
    CHECK(!i2.in(3.0));
  }

  TEST(greater)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(4.0, 1.0, excluded, included);
    interval<double> i2(7.0, 8.0, included, included);

    CHECK(i0.greater(3.0));
    CHECK(i0.greater(5.0));
    CHECK(!i0.greater(2.3));
    
    CHECK(i1.greater(4.0));
    CHECK(i1.greater(5.0));
    CHECK(!i1.greater(3.3));

    CHECK(!i2.greater(4.0));
    CHECK(!i2.greater(8.0));
    CHECK(i2.greater(8.3));
  }

  TEST(less)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(4.0, 1.0, excluded, included);
    interval<double> i2(7.0, 8.0, included, included);

    CHECK(i0.less(2.0));
    CHECK(i0.less(0.0));
    CHECK(!i0.less(2.3));
    
    CHECK(!i1.less(1.0));
    CHECK(i1.less(0.0));
    CHECK(!i1.less(3.3));

    CHECK(i2.less(4.0));
    CHECK(!i2.less(7.0));
    CHECK(!i2.less(8.3));
  }

  TEST(merge)
  {
    interval<double> i0(2.0, 3.0, excluded, excluded);
    interval<double> i1(4.0, 1.0, excluded, included);
    interval<double> i3(7.0, 8.0, included, included);
  }

  TEST(nil)
  {
    interval<double> i0(2.0, 2.0, excluded, excluded);
    interval<double> i1(2.0, 2.0, excluded, included);
    interval<double> i2(2.0, 2.0, included, excluded);
    interval<double> i3(2.0, 2.0, included, included);

    CHECK(i0.nil());
    CHECK(i1.nil());
    CHECK(i2.nil());
    CHECK(!i3.nil());
  }

  TEST(print)
  {}
}
