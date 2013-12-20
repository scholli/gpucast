/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testnurbssurface.cpp
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
#include <set>

#include <gpucast/math/parametric/nurbssurface.hpp>

using namespace gpucast::math;

SUITE (nurbssurface_class)
{
  TEST(default_ctor)
  {
    nurbssurface3d ns;
  }


  TEST(swap)
  {
    nurbssurface3d ns;
    nurbssurface3d ns2;
    ns.swap(ns2);
  }

  TEST(verify)
  {
    nurbssurface3d ns;

    ns.degree_u(2);
    ns.degree_v(2);

    CHECK(!ns.verify());
  }

  TEST(order_degree)
  {
    nurbssurface3d ns;

    ns.degree_u(2);
    ns.degree_v(2);

    CHECK(!ns.verify());
    CHECK(ns.order_u() == 3);
    CHECK(ns.order_v() == 3);
  }

  TEST(knotvector_u)
  {
    double k[] = {0, 0, 0, 0, 0.3, 1.0, 1.0, 1.0, 1.0};
    nurbssurface3d ns;

    ns.knotvector_u(k, k+9);
    CHECK(ns.knotvector_u().size() == 9);
    CHECK(ns.knotvector_v().size() == 0);
  }

  TEST(knotvector_v)
  {
    double k[] = {0, 0, 0, 0, 0.3, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0};
    nurbssurface3d ns;

    ns.knotvector_v(k, k+11);
    CHECK(ns.knotvector_v().size() == 11);
    CHECK(ns.knotvector_u().size() == 0);
  }

  TEST(set_points)
  {
    std::vector<point3d> pts(12);
    nurbssurface3d ns;
    ns.set_points(pts.begin(), pts.end());
    CHECK(ns.points().size() == 12);
  }

  TEST(numberofpoints)
  {
    std::vector<point3d> pts(20);
    nurbssurface3d ns;
    ns.set_points(pts.begin(), pts.end());

    ns.numberofpoints_u(4);
    ns.numberofpoints_v(5);

    ns.degree_u(2);
    ns.degree_v(2);

    std::vector<double> ku(7);
    std::vector<double> kv(8);

    ns.knotvector_u(ku.begin(), ku.end());
    ns.knotvector_v(kv.begin(), kv.end());

    CHECK(ns.numberofpoints_u() == 4);
    CHECK(ns.numberofpoints_v() == 5);
    CHECK(ns.verify());
  }

}

