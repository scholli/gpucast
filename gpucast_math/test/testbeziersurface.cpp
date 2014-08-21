/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testbeziersurface.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <gpucast/math/parametric/beziersurface.hpp>

using namespace gpucast::math;

SUITE (beziersurface_class)
{

  beziersurface3d create_beziersurface()
  {
    beziersurface3d bs;

    bs.degree_u(2);
    bs.degree_v(3);

    bs.add(point3d(0.0, 0.0, 0.0, 1.0));
    bs.add(point3d(0.1, 1.0, 2.0, 3.0));
    bs.add(point3d(0.2, 2.2, 1.0, 1.0));
    bs.add(point3d(0.0, 3.0, 1.0, 1.0));

    bs.add(point3d(1.0, 0.0, 3.0, 2.0));
    bs.add(point3d(1.0, 1.1, 1.0, 3.0));
    bs.add(point3d(1.3, 2.5, 2.0, 1.0));
    bs.add(point3d(1.0, 3.4, 4.0, 1.0));

    bs.add(point3d(2.0, 0.0, 2.0, 1.0));
    bs.add(point3d(2.3, 1.0, 4.0, 2.0));
    bs.add(point3d(2.1, 2.3, 0.0, 1.0));
    bs.add(point3d(2.0, 3.1, 1.0, 1.0));

    return bs;
  }

  TEST(ctor)
  {
    beziersurface3d bs;

    CHECK(bs.size() == 0);

    bs = create_beziersurface();

    CHECK(bs.size() == 12);
  }

  TEST(swap)
  {
    beziersurface3d bs = create_beziersurface();
    beziersurface3d bs2;

    bs.swap(bs2);
    CHECK(bs.size() == 0);
    CHECK(bs.degree_u() == 0);
    CHECK(bs.degree_v() == 0);

    CHECK(bs2.size() == 12);
    CHECK(bs2.degree_u() == 2);
    CHECK(bs2.degree_v() == 3);
  }

  TEST(assignmentoperator)
  {
    beziersurface3d bs = create_beziersurface();
    CHECK(bs.size() == 12);
    CHECK(bs.degree_u() == 2);
    CHECK(bs.degree_v() == 3);
  }

  TEST(set_points)
  {
    beziersurface3d bs;
    std::vector<point3d> points(16);
    bs.mesh(points.begin(), points.end());

    CHECK(bs.size() == 16);
  }

  TEST(degree_order)
  {
    beziersurface3d bs;

    bs.degree_u(4);
    bs.degree_v(5);

    CHECK(bs.degree_u() == 4);
    CHECK(bs.degree_v() == 5);
    CHECK(bs.order_u() == 5);
    CHECK(bs.order_v() == 6);
  }

  TEST(elevate)
  {
    beziersurface3d bs = create_beziersurface();
    beziersurface3d tmp = bs;
    bs.elevate_u();

    CHECK(bs.size() == 16);
    CHECK(bs.degree_u() == 3);
    CHECK(bs.degree_v() == 3);

    bs.elevate_v();
    CHECK(bs.size() == 20);
    CHECK(bs.degree_v() == 4);

    horner<point3d> h3d;

    point3d a = bs.evaluate(0.3, 0.4, h3d);
    point3d b = tmp.evaluate(0.3, 0.4, h3d);

    CHECK_CLOSE(a.distance(b), 0.0, 0.000001);
  }

  TEST(evaluate)
  {
    beziersurface3d bs = create_beziersurface();

    horner<point3d> h3d;
    decasteljau<point3d> d3d;
    point3d du0, du1, dv0, dv1, p0, p1, p2, p3, p4;

    p0 = bs.evaluate(0.3, 0.4, h3d);

    bs.evaluate(0.3, 0.4, p1, du0, dv0, h3d);
    bs.evaluate(0.3, 0.4, p4, du1, dv1, d3d);

    bs.evaluate(0.3, 0.4, p2, h3d);
    bs.evaluate(0.3, 0.4, p3, d3d);

    CHECK_CLOSE(p3.distance(p2), 0.0, 0.000001);
    CHECK_CLOSE(p0.distance(p1), 0.0, 0.000001);
    CHECK_CLOSE(p2.distance(p1), 0.0, 0.000001);
    CHECK_CLOSE(p4.distance(p1), 0.0, 0.000001);

    CHECK_CLOSE(du0.distance(du1), 0.0, 0.000001);
    CHECK_CLOSE(dv0.distance(dv1), 0.0, 0.000001);
  }

  TEST(curvature)
  {
    beziersurface3d bs = create_beziersurface();

    // just instanciate
    bs.curvature();
  }

  TEST(split)
  {
    beziersurface3d bs = create_beziersurface();
    std::vector<beziersurface3d> splitlist;
    bs.split(std::back_inserter(splitlist));

    CHECK(splitlist.size() == 4);

    for (std::vector<beziersurface3d>::const_iterator i = splitlist.begin(); i != splitlist.end(); ++i)
    {
      CHECK(i->order_u() == bs.order_u());
      CHECK(i->order_v() == bs.order_v());
    }

    double original_curvature = bs.curvature();
    std::set<double> curvatures;
    std::transform(splitlist.begin(), splitlist.end(), std::inserter(curvatures, curvatures.begin()), [&] (beziersurface3d const& bs) { return bs.curvature(); } );

    for (double const& curvature : curvatures)
    {
      CHECK(curvature <= original_curvature);
    }
  }


  TEST(bbox)
  {
    beziersurface3d bs = create_beziersurface();

    beziersurface3d::bbox_t bb = bs.bbox();
  }

  TEST(print)
  {

  }


}
