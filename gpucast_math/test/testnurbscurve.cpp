/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testnurbscurve.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <iostream>
#include <set>

#include <unittest++/UnitTest++.h>

#include <gpucast/math/parametric/nurbscurve.hpp>

using namespace gpucast::math;

SUITE (nurbscurve_class)
{

  nurbscurve2d create_testcurve()
  {
    nurbscurve2d nc;

    nc.add(point2d( 1, 3, 2));
    nc.add(point2d(-3, 2, 4));
    nc.add(point2d(-4, 5, 1));
    nc.add(point2d( 0, 8, 4));
    nc.add(point2d( 5, 4, 2));
    nc.add(point2d( 1, 2, 3));
    nc.add(point2d( 3, 0, 4));
    nc.add(point2d( 6, 2, 2));

    std::vector<double> knots;

    nc.add_knot(0);
    nc.add_knot(0);
    nc.add_knot(0);
    nc.add_knot(0);
    nc.add_knot(1);
    nc.add_knot(3);
    nc.add_knot(3);
    nc.add_knot(4);
    nc.add_knot(6);
    nc.add_knot(6);
    nc.add_knot(6);
    nc.add_knot(6);

    nc.degree(3);
    return nc;
  }

  TEST(ctor)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.verify());
  }

  TEST(swap)
  {
    nurbscurve2d nc = create_testcurve();
    nurbscurve2d nc2 = nc;
    nc2[0] = point2d(1,1,1);

    nc.swap(nc2);

    CHECK(nc[0] == point2d(1,1,1));
    CHECK(nc2[0] == point2d(1,3,2));
  }

  TEST(print)
  {
    nurbscurve2d nc = create_testcurve();
  }

  TEST(add)
  {
    nurbscurve2d nc = create_testcurve();
    nc.add(point2d(1,4,4));
    CHECK(nc[nc.size()-1] == point2d(1,4,4));
  }

  TEST(add_knot)
  {
    nurbscurve2d nc = create_testcurve();
    nc.add_knot(6.0);
  }

  TEST(add_knot2)
  {
    nurbscurve2d nc = create_testcurve();
    nc.add_knot(0.0, 0);
  }

  TEST(clear)
  {
    nurbscurve2d nc = create_testcurve();
    nc.clear();

    CHECK(nc.size() == 0);
    CHECK(nc.knots().size() == 0);
  }

  TEST(verify)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.verify());

    nc.degree(4);
    CHECK(!nc.verify());
  }

  TEST(degree)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.degree() == 3);
  }

  TEST(order)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.order() == 4);
  }

  TEST(size)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.size() == 8);
  }

  TEST(normalize_knotvector)
  {
    nurbscurve2d nc = create_testcurve();
    nc.normalize_knotvector();
    std::set<point2d::value_type> knots(nc.knots().begin(), nc.knots().end());

    CHECK(knots.find(0.0) != knots.end());
    CHECK(knots.find(1.0) != knots.end());
    CHECK(knots.find(1.0/6.0) != knots.end());
    CHECK(knots.find(3.0/6.0) != knots.end());
    CHECK(knots.find(4.0/6.0) != knots.end());
    CHECK(knots.find(3.0) == knots.end());
    CHECK(knots.find(4.0) == knots.end());
    CHECK(knots.find(6.0) == knots.end());

  }

  TEST(set_knotvector)
  {
    nurbscurve2d nc = create_testcurve();
    double knots[] = {0, 0 ,0, 0, 2, 3, 3, 4, 5, 5, 5, 5};
    nc.set_knotvector(knots, knots + 12);

    CHECK(nc.knots()[0] == 0);
    CHECK(nc.knots()[1] == 0);
    CHECK(nc.knots()[2] == 0);
    CHECK(nc.knots()[3] == 0);
    CHECK(nc.knots()[4] == 2.0/5.0);
    CHECK(nc.knots()[5] == 3.0/5.0);
    CHECK(nc.knots()[6] == 3.0/5.0);
    CHECK(nc.knots()[7] == 4.0/5.0);
    CHECK(nc.knots()[8] == 1.0);
    CHECK(nc.knots()[9] == 1.0);
    CHECK(nc.knots()[10] == 1.0);
    CHECK(nc.knots()[11] == 1.0);
  }

  TEST(set_points)
  {
    std::vector<point2d> tmp(8);

    nurbscurve2d nc = create_testcurve();

    nc.set_points(tmp.begin(), tmp.end());
    CHECK(nc[0] == point2d());
    CHECK(nc[1] == point2d());
    CHECK(nc[2] == point2d());
    CHECK(nc[3] == point2d());
    CHECK(nc[4] == point2d());
    CHECK(nc[5] == point2d());
    CHECK(nc[6] == point2d());
    CHECK(nc[7] == point2d());
  }

  TEST(points)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.points().size() == 8);
  }

  TEST(knots)
  {
    nurbscurve2d nc = create_testcurve();
    CHECK(nc.knots().size() == 12);
  }

}
