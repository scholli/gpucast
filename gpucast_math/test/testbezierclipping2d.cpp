/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testbezierclipping.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <cstdlib>

#include <gpucast/math/parametric/algorithm/bezierclipping2d.hpp>
#include <gpucast/math/parametric/beziercurve.hpp>


using namespace gpucast::math;

SUITE (bezierclipping2d)
{

  beziercurve2d create_circle()
  {
    // create a unit circle as rational bezier curve
    beziercurve2d bc;

    point2d p0( 0.0, -1.0, 1.0);
    point2d p1( 4.0, -1.0, 1.0/5.0);
    point2d p2( 2.0,  3.0, 1.0/5.0);
    point2d p3(-2.0,  3.0, 1.0/5.0);
    point2d p4(-4.0, -1.0, 1.0/5.0);
    point2d p5( 0.0, -1.0, 1.0);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);
    bc.add(p3);
    bc.add(p4);
    bc.add(p5);

    return bc;
  }

  TEST(clip)
  {
    beziercurve2d c = create_circle();
    bezierclipping2d<point2d> bclip;

    unsigned const nrandon_points = 10000;

    for (std::size_t i = 0; i != nrandon_points; ++i)
    {
      int xint = rand();
      int yint = rand();

      double x = double(2*xint)/RAND_MAX;
      double y = double(2*yint)/RAND_MAX;

      point2d p(x,y);
      bool inside = p.distance(point2d(0.0, 0.0)) <= 1.0;

      std::size_t iters = 0;
      CHECK(inside == bclip.intersects_odd_times(c, p, point2d::u, 64, iters));
      CHECK(inside == bclip.intersects_odd_times(c, p, point2d::v, 64, iters));
    }
  }
}
