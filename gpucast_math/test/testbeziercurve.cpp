/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testbeziercurve.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <vector>
#include <memory>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/parametric/algorithm/decasteljau.hpp>
#include <gpucast/math/axis_aligned_boundingbox.hpp>

using namespace gpucast::math;

namespace
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
}

SUITE (beziercurve_classes)
{

  TEST(add)
  {
    beziercurve2d bc;

    point2d p0(1.0, 0.0);
    point2d p1(3.0, 0.0);
    point2d p2(5.0, 0.0);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);

    CHECK(bc.degree() == 2);

    CHECK(bc[0] == p0);
    CHECK(bc[1] == p1);
    CHECK(bc[2] == p2);
  }

  TEST(evaluate)
  {
    horner<point2f> h;

    beziercurve2f bc;

    point2f p(2.0f, 2.0f);
    point2f p0(3.0f, 2.0f, 1.0f);
    point2f p1(3.0f, 3.0f, sqrt(2.0f)/2.0f);
    point2f p2(2.0f, 3.0f, 1.0f);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);

    point2f p3;
    point2f p4;

    bc.evaluate(0.2f, p3, h);
    bc.evaluate(0.6f, p4, h);
    point2f p5 = bc.evaluate(0.2f, h);

    float len3 = sqrt((p3[0]-p[0]) * (p3[0]-p[0]) + (p3[1]-p[1]) * (p3[1]-p[1]));
    float len4 = sqrt((p4[0]-p[0]) * (p4[0]-p[0]) + (p4[1]-p[1]) * (p4[1]-p[1]));

    CHECK_CLOSE(len3, 1.0, 0.00001);
    CHECK_CLOSE(len4, 1.0, 0.00001);
    CHECK(p3 == p5);

    decasteljau<point2f> d2f;

    point2f d3, d4;
    bc.evaluate(0.2f, d3, d2f);
    bc.evaluate(0.6f, d4, d2f);

    CHECK_CLOSE(p3.distance(d3), 0.0, 0.00001);
    CHECK_CLOSE(p4.distance(d4), 0.0, 0.00001);
  }

  TEST(scaled_hodograph)
  {
    horner<point2f> h;

    beziercurve2f bc;

    point2f p0(3.0f, 2.0f, 1.0f);
    point2f p1(3.0f, 3.0f, sqrt(2.0f)/2.0f);
    point2f p2(2.0f, 3.0f, 1.0f);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);

    beziercurve2f sc = bc.scaled_hodograph();

    point2f p;
    point2f pt;
    point2f dt;

    sc.evaluate(0.3f, p, h);
    bc.evaluate(0.3f, pt, dt, h);
    CHECK_CLOSE(dt[1]/dt[0], p[1]/p[0], 0.00001);

    sc.evaluate(0.6f, p, h);
    bc.evaluate(0.6f, pt, dt, h);
    CHECK_CLOSE(dt[1]/dt[0], p[1]/p[0], 0.00001);

    sc.evaluate(0.7f, p, h);
    bc.evaluate(0.7f, pt, dt, h);
    CHECK_CLOSE(dt[1]/dt[0], p[1]/p[0], 0.00001);
  }

  TEST(hodograph)
  {
    horner<point2f> h;

    beziercurve2f bc;

    point2f p0(3.0f, 2.0f, 1.0f);
    point2f p1(3.0f, 3.0f, sqrt(2.0f)/2.0f);
    point2f p2(2.0f, 3.0f, 1.0f);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);

    point2f p;
    point2f dt;

    bc.evaluate(0.5f, p, dt, h);
    CHECK_CLOSE(dt[1]/dt[0], -1.0, 0.0001);
  }

  TEST(nishita)
  {
    beziercurve3d bc;

    point3d p0(2.0, -2.0,  1.0);
    point3d p1(4.0,  5.0,  2.0);
    point3d p2(7.0,  1.0, -1.0);
    point3d p3(8.0,  3.0,  4.0);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);
    bc.add(p3);

    beziercurve2d bc0 = bc.nishita(0);
    beziercurve2d bc1 = bc.nishita(1);
    beziercurve2d bc2 = bc.nishita(2);

    CHECK_EQUAL(bc0[0], point2d(0.0/3.0,  2.0f));
    CHECK_EQUAL(bc0[1], point2d(1.0/3.0,  4.0f));
    CHECK_EQUAL(bc0[2], point2d(2.0/3.0,  7.0f));
    CHECK_EQUAL(bc0[3], point2d(3.0/3.0,  8.0f));

    CHECK_EQUAL(bc1[0], point2d(0.0/3.0, -2.0f));
    CHECK_EQUAL(bc1[1], point2d(1.0/3.0,  5.0f));
    CHECK_EQUAL(bc1[2], point2d(2.0/3.0,  1.0f));
    CHECK_EQUAL(bc1[3], point2d(3.0/3.0,  3.0f));

    CHECK_EQUAL(bc2[0], point2d(0.0/3.0,  1.0f));
    CHECK_EQUAL(bc2[1], point2d(1.0/3.0,  2.0f));
    CHECK_EQUAL(bc2[2], point2d(2.0/3.0, -1.0f));
    CHECK_EQUAL(bc2[3], point2d(3.0/3.0,  4.0f));
  }

  TEST(rational)
  {
    beziercurve2f bc;

    point2f p0(2.0f, -2.0f);
    point2f p1(4.0f,  5.0f);
    point2f p2(7.0f,  1.0f);
    point2f p3(8.0f,  3.0f);

    bc.add(p0);
    bc.add(p1);
    bc.add(p2);
    bc.add(p3);

    CHECK(bc.is_rational() == false);

    bc[2] = point2f(7.0f, 1.0f, 2.0f);

    CHECK(bc.is_rational() == true);
  }

  TEST(clip_left_and_right)
  {
    beziercurve2d bc = create_circle();
    horner<point2d> h2d;

    beziercurve2d bc2 = bc;
    beziercurve2d bc3 = bc;

    // clip some thing left and right points should have abs of 1
    bc.clip_left(0.3f);
    bc.clip_left(0.1f);

    CHECK_CLOSE(bc.evaluate(0.0f, h2d).abs(), 1.0, 0.00001);
    CHECK_CLOSE(bc.evaluate(0.3f, h2d).abs(), 1.0, 0.00001);
    CHECK_CLOSE(bc.evaluate(0.4f, h2d).abs(), 1.0, 0.00001);
    CHECK_CLOSE(bc.evaluate(0.6f, h2d).abs(), 1.0, 0.00001);

    bc2.clip_left(0.5f);
    bc3.clip_right(0.5f);

    CHECK_CLOSE(bc2[0].distance(point2d(0.0f, 1.0f)), 0.0, 0.0001);
    CHECK_CLOSE(bc3[5].distance(point2d(0.0f, 1.0f)), 0.0, 0.0001);
    CHECK_CLOSE(bc2[5].distance(point2d(0.0f, -1.0f)), 0.0, 0.0001);
    CHECK_CLOSE(bc3[0].distance(point2d(0.0f, -1.0f)), 0.0, 0.0001);
  }

  TEST(clip_lr)
  {
    beziercurve2d bc = create_circle();

    bc.clip_lr(1.0/2.0, 1.0/2.0);

    CHECK_CLOSE(bc[0].distance(point2d(0.0, 1.0)), 0.0, 0.0001);
    CHECK_CLOSE(bc[1].distance(point2d(0.0, 1.0)), 0.0, 0.0001);
    CHECK_CLOSE(bc[2].distance(point2d(0.0, 1.0)), 0.0, 0.0001);
    CHECK_CLOSE(bc[3].distance(point2d(0.0, 1.0)), 0.0, 0.0001);
    CHECK_CLOSE(bc[4].distance(point2d(0.0, 1.0)), 0.0, 0.0001);
    CHECK_CLOSE(bc[5].distance(point2d(0.0, 1.0)), 0.0, 0.0001);
  }

  TEST(split)
  {
    beziercurve2d bc = create_circle();

    beziercurve2d bc2;
    beziercurve2d bc3;
    bc.split(0.5f, bc2, bc3);

    CHECK_CLOSE(bc2[5].distance(point2d(0.0, 1.0)), 0.0, 0.00001);
    CHECK_CLOSE(bc3[0].distance(point2d(0.0, 1.0)), 0.0, 0.00001);
    CHECK_CLOSE(bc2[0].distance(point2d(0.0f, -1.0f)), 0.0, 0.0001);
    CHECK_CLOSE(bc3[5].distance(point2d(0.0f, -1.0f)), 0.0, 0.0001);
  }

  TEST(split2)
  {
    beziercurve2d bc = create_circle();

    std::set<double> splits;

    splits.insert(0.3);
    splits.insert(0.4);
    splits.insert(0.6);
    splits.insert(0.7);
    splits.insert(0.9);

    std::vector<beziercurve2d> split_container;

    bc.split(splits, split_container);

    CHECK( splits.size()+1 == split_container.size() );

    CHECK( split_container.front().front()  == bc.front() );
    CHECK( split_container.back().back()    == bc.back() );

    for (std::size_t i = 0; i != split_container.size(); ++i )
    {
      if ( i+1 < split_container.size() )
      {
        CHECK( split_container.at(i).back() == split_container.at(i+1).front() );
      }
    }
  }


  TEST(extrema)
  {
    beziercurve2d bc = create_circle();

    std::set<point2d::value_type> roots;

    bc.extrema(0, roots, 128);
    bc.extrema(1, roots, 128);

    //std::copy(roots.begin(), roots.end(), std::ostream_iterator<point2d::value_type>(std::cout, " "));
    CHECK(roots.size() == 5);

    std::vector<point2d::value_type> rootvec(roots.begin(), roots.end());

    CHECK_CLOSE(rootvec[0],                 0.0, 0.0000001);
    CHECK_CLOSE(rootvec[1], 1.0 - sqrt(2.0)/2.0, 0.0000001);
    CHECK_CLOSE(rootvec[2],                0.5f, 0.0000001);
    CHECK_CLOSE(rootvec[3],       sqrt(2.0)/2.0, 0.0000001);
    CHECK_CLOSE(rootvec[4],                 1.0, 0.0000001);

    decasteljau<point2d> d2d;

    point2d p0, p1, p2, p3, p4;
    point2d d0, d1, d2, d3, d4;

    bc.evaluate(rootvec[0], p0, d0, d2d);
    bc.evaluate(rootvec[1], p1, d1, d2d);
    bc.evaluate(rootvec[2], p2, d2, d2d);
    bc.evaluate(rootvec[3], p3, d4, d2d);
    bc.evaluate(rootvec[4], p3, d4, d2d);

    CHECK_CLOSE(d0[1], 0.0, 0.000001);
    CHECK_CLOSE(d1[0], 0.0, 0.000001);
    CHECK_CLOSE(d2[1], 0.0, 0.000001);
    CHECK_CLOSE(d3[0], 0.0, 0.000001);
    CHECK_CLOSE(d4[1], 0.0, 0.000001);
  }


  TEST(extrema2)
  {
    std::shared_ptr<gpucast::math::beziercurve2d> bc(new gpucast::math::beziercurve2d);

    gpucast::math::point2d p0(-0.8f, 0.7f);
    gpucast::math::point2d p1(-0.4f, -0.8f);
    gpucast::math::point2d p2(-0.2f, 0.1f);
    gpucast::math::point2d p3(0.7f, 0.7f);
    gpucast::math::point2d p4(0.9f, -0.6f);

    bc->add(p0);
    bc->add(p1);
    bc->add(p2);
    bc->add(p3);
    bc->add(p4);

    std::set<double> extrema;
    bc->extrema(point2d::u, extrema, 64);
    bc->extrema(point2d::v, extrema, 64);

    CHECK(extrema.size() == 2);
  }


  TEST(bisect)
  {
    beziercurve2d bc = create_circle();

    bool is_root(false);
    point2d::value_type root_parameter(0.0);

    bc.bisect( 1, 0.0, is_root, root_parameter, interval<point2d::value_type>(0.0, 0.5), 128);

    CHECK(is_root);
    CHECK_CLOSE(root_parameter, 1.0 - sqrt(2.0)/2.0, 0.00001);

    bc.bisect( 1,-1.0, is_root, root_parameter, interval<point2d::value_type>(0.0, 0.5), 128);

    CHECK(is_root);
    CHECK_CLOSE(root_parameter, 0.0, 0.0);

    bc.bisect( 1, 1.0, is_root, root_parameter, interval<point2d::value_type>(0.0, 0.5), 128);

    CHECK(is_root);
    CHECK_CLOSE(root_parameter, 0.5, 0.000001);

    bc.bisect( 1, 0.0, is_root, root_parameter, interval<point2d::value_type>(0.5, 1.0), 128);

    CHECK(is_root);
    CHECK_CLOSE(root_parameter, sqrt(2.0)/2.0, 0.00001);

    bc.bisect( 1, 1.2, is_root, root_parameter, interval<point2d::value_type>(0.5, 1.0), 128);
    CHECK(!is_root);
  }

  TEST(optbisect)
  {
    beziercurve2d bc = create_circle();

    bool is_root(false);
    std::size_t iters(0);

    bc.optbisect(point2d(0.0, 0.0), 1, 0, is_root, iters, interval<point2d::value_type>(0.0, 0.5), 128);
    CHECK(is_root);

    bc.optbisect(point2d(0.0, 0.0), 1, 0, is_root, iters, interval<point2d::value_type>(0.0, 1.0 - sqrt(2.0)/2.0 + 0.00001), 128);
    CHECK(is_root);

    bc.optbisect(point2d(0.0, 0.0), 1, 0, is_root, iters, interval<point2d::value_type>(0.0, 1.0 - sqrt(2.0)/2.0 - 0.00001), 128);
    CHECK(!is_root);
  }

  TEST(minimum)
  {
    beziercurve2d bc = create_circle();
    CHECK_EQUAL(bc.minimum(0), -4);
    CHECK_EQUAL(bc.minimum(1), -1);
  }

  TEST(maximum)
  {
    beziercurve2d bc = create_circle();
    CHECK_EQUAL(bc.maximum(0), 4);
    CHECK_EQUAL(bc.maximum(1), 3);
  }

  TEST(is_linear)
  {
    beziercurve2d bc = create_circle();
    CHECK(bc.is_constant(0) == false);
    CHECK(bc.is_constant(1) == false);

    beziercurve2d bc2;
    bc2.add(point2d(0.0, 1.0));
    bc2.add(point2d(4.0, 1.0));
    CHECK(bc2.is_constant(0) == false);
    CHECK(bc2.is_constant(1) == true);

    bc2.add(point2d(6.0, 1.01));
    CHECK(bc2.is_constant(0) == false);
    CHECK(bc2.is_constant(1) == false);
    CHECK(bc2.is_constant(1, 0.02) == true);

    beziercurve2d bc3;
    bc3.add(point2d(4.0, 0.0));
    bc3.add(point2d(4.0, 1.0));
    CHECK(bc3.is_constant(1) == false);
    CHECK(bc3.is_constant(0) == true);
  }

  TEST(elevate)
  {
    beziercurve2d bc = create_circle();
    beziercurve2d bc2 = bc;

    bc2.elevate();

    horner<point2d> h2d;

    point2d a0, da0, a1, da1;
    point2d b0, db0, b1, db1;

    bc.evaluate(0.3, a0, da0, h2d);
    bc2.evaluate(0.3, a1, da1, h2d);

    bc.evaluate(0.4, b0, db0, h2d);
    bc2.evaluate(0.4, b1, db1, h2d);

    CHECK_CLOSE(a0.distance(a1), 0.0, 0.0001);
    CHECK_CLOSE(b0.distance(b1), 0.0, 0.0001);
    CHECK_CLOSE(da0.distance(da1), 0.0, 0.0001);
    CHECK_CLOSE(db0.distance(db1), 0.0, 0.0001);
  }

  TEST(translate)
  {
    beziercurve2d bc = create_circle();

    bc.translate(point2d(2.0, 2.0));

    CHECK_EQUAL(bc[0], point2d( 2.0,  1.0, 1.0));
    CHECK_EQUAL(bc[1], point2d( 6.0,  1.0, 1.0/5.0));
    CHECK_EQUAL(bc[2], point2d( 4.0,  5.0, 1.0/5.0));
    CHECK_EQUAL(bc[3], point2d( 0.0,  5.0, 1.0/5.0));
    CHECK_EQUAL(bc[4], point2d(-2.0,  1.0, 1.0/5.0));
    CHECK_EQUAL(bc[5], point2d( 2.0,  1.0, 1.0));
  }

  TEST(bbox2D_simple)
  {
    beziercurve2d bc = create_circle();
    axis_aligned_boundingbox<point2d> bb;

    bc.bbox_simple(bb);

    CHECK(bb.min == point2d(-4.0, -1.0));
    CHECK(bb.max == point2d( 4.0,  3.0));
  }


  TEST(bbox2D_tight)
  {
    beziercurve2d bc = create_circle();

    axis_aligned_boundingbox<point2d> bb;

    bc.bbox_tight(bb);
    CHECK_CLOSE(bb.min.distance(point2d(-1.0, -1.0)), 0.0, 0.00001);
    CHECK_CLOSE(bb.max.distance(point2d( 1.0,  1.0)), 0.0, 0.00001);
  }

  TEST(front)
  {
    beziercurve2d bc = create_circle();
    CHECK_EQUAL(point2d(0.0, -1.0, 1.0), bc.front());

    beziercurve2d bc2;
    bc2.add(point2d(0.0, 1.0));
    bc2.add(point2d(4.0, 1.0));
    bc2.add(point2d(6.0, 1.01));
    CHECK_EQUAL(point2d(0.0, 1.0), bc2.front());
  }
  TEST(back)
  {
    beziercurve2d bc = create_circle();
    CHECK_EQUAL(point2d(0.0, -1.0, 1.0), bc.back());

    beziercurve2d bc2;
    bc2.add(point2d(0.0, 1.0));
    bc2.add(point2d(4.0, 1.0));
    bc2.add(point2d(6.0, 1.01));
    CHECK_EQUAL(point2d(6.0, 1.01), bc2.back());
  }

  TEST(invert)
  {
    beziercurve2d bc2;

    bc2.add(point2d(0.0, 1.0));
    bc2.add(point2d(4.0, 1.0));
    bc2.add(point2d(6.0, 1.01));

    bc2.invert();

    CHECK_EQUAL(point2d(6.0, 1.01), bc2[0]);
    CHECK_EQUAL(point2d(4.0, 1.0), bc2[1]);
    CHECK_EQUAL(point2d(0.0, 1.0), bc2[2]);
  }

  TEST(is_increasing)
  {
    beziercurve2d bc2;

    bc2.add(point2d(0.0, 1.0));
    bc2.add(point2d(4.0, 1.0));
    bc2.add(point2d(6.0, 1.01));

    CHECK(bc2.is_increasing(1));
    CHECK(bc2.is_increasing(0));

    bc2.invert();
    CHECK(!bc2.is_increasing(1));
    CHECK(!bc2.is_increasing(0));

    bc2.invert();

    bc2.add(point2d(-1.0, 1.5));
    CHECK(bc2.is_increasing(1));
    CHECK(!bc2.is_increasing(0));

    bc2.add(point2d(1.0, 0.5));
    CHECK(!bc2.is_increasing(1));
    CHECK(bc2.is_increasing(0));
  }

  TEST(clear)
  {
    beziercurve2d bc = create_circle();
    bc.clear();
    CHECK_EQUAL(bc.degree(), 0);
  }

  TEST(decasteljau)
  {
    beziercurve2d bc = create_circle();

    point2d p, dt;
    bc.evaluate(0.3, p, dt, decasteljau<point2d>());

    CHECK_CLOSE(dt[0], -0.336217147074502, 0.000000000001);
    CHECK_CLOSE(dt[1], 6.888351305916603, 0.000000000001);
    CHECK_CLOSE(dt.weight(), -0.663878620689655, 0.000000000001);

    CHECK_CLOSE(0.998810939357907, p[0], 0.00000000001);
    CHECK_CLOSE(0.048751486325802, p[1], 0.00000000001);
    CHECK_CLOSE(0.336400000000000, p.weight(), 0.00000000001);

    point2d p1, p2, dt1, dt2;

    bc.evaluate(1.0, p1, dt1, decasteljau<point2d>());
    bc.evaluate(0.0, p2, dt2, decasteljau<point2d>());

    CHECK_CLOSE( 0.0, p1[0], 1.0);
    CHECK_CLOSE(-1.0, p1[1], 1.0);
    CHECK_CLOSE( 0.0, p1.weight(), 1.0);

    CHECK_CLOSE( 0.0, p2[0], 1.0);
    CHECK_CLOSE(-1.0, p2[1], 1.0);
    CHECK_CLOSE( 0.0, p2.weight(), 1.0);
  }

  TEST(decasteljau_start_end_point_robustness)
  {
    beziercurve2d bc = create_circle();
    point2d p1, p2, dt1, dt2;

    bc.evaluate(1.0, p1, dt1, decasteljau<point2d>());
    bc.evaluate(0.0, p2, dt2, decasteljau<point2d>());

    CHECK_CLOSE( 0.0, p1[0], 1.0);
    CHECK_CLOSE(-1.0, p1[1], 1.0);
    CHECK_CLOSE( 0.0, p1.weight(), 1.0);

    CHECK_CLOSE( 0.0, p2[0], 1.0);
    CHECK_CLOSE(-1.0, p2[1], 1.0);
    CHECK_CLOSE( 0.0, p2.weight(), 1.0);
  }

  TEST(weak_monotonic)
  {
    beziercurve2d bc1, bc2, bc3;
    bc1.add(point2d(542, -28));
    bc1.add(point2d(516, -28));
    bc1.add(point2d(580,   2));

    bc2.add(point2d(571, 1275));
    bc2.add(point2d(614, 1291));
    bc2.add(point2d(640, 1291));

    bc3.add(point2d(442, -28));
    bc3.add(point2d(516,  28));
    bc3.add(point2d(580,   2));


    CHECK(!bc1.weak_monotonic(0));
    CHECK(bc2.weak_monotonic(0));
    CHECK(bc3.weak_monotonic(0));

    CHECK(bc1.weak_monotonic(1));
    CHECK(bc2.weak_monotonic(1));
    CHECK(!bc3.weak_monotonic(1));
  }

  TEST(curvature)
  {
    beziercurve2d bc1, bc2, bc3;
    bc1.add(point2d(542, -28));
    bc1.add(point2d(516, -28));
    bc1.add(point2d(580,   2));

    bc2.add(point2d(571, 1291));
    bc2.add(point2d(614, 1291));
    bc2.add(point2d(640, 1291));

    bc3.add(point2d(442, -28));
    bc3.add(point2d(516,  28));
    bc3.add(point2d(580,   2));

    double const eps = 0.0001;
    CHECK       ( bc1.curvature() > 0.1 );
    CHECK_CLOSE ( 0.0, bc2.curvature(), eps);
    CHECK       ( bc3.curvature() > 0.1 );
  }

  TEST(bcurve_polynomial)
  {
    beziercurve2d bc1;
    bc1.add(point2d(5, 7 ));
    bc1.add(point2d(7, 12));
    bc1.add(point2d(8, 13));

    polynomial<double> upoly = bc1.as_polynomial(point2d::u);
    polynomial<double> vpoly = bc1.as_polynomial(point2d::v);

    CHECK_CLOSE ( upoly.evaluate(0.3), bc1.evaluate(0.3)[point2d::u], 0.00001 );
    CHECK_CLOSE ( vpoly.evaluate(0.3), bc1.evaluate(0.3)[point2d::v], 0.00001 );
  }

  TEST(partition_optimization)
  {
    beziercurve2d bc1, bc2;
    bc1.add(point2d(5, 7 ));
    bc1.add(point2d(7, 12));
    bc1.add(point2d(8, 13));

    beziercurve2d tbc1 = bc1;

    // translate into origin
    if ( bc1.is_increasing(point2d::u) )
    {
      tbc1.translate(-tbc1.front());
    } else {
      tbc1.translate(-tbc1.back());
    }

    // compute bounds
    axis_aligned_boundingbox<point2d> bb1, bb2;
    tbc1.bbox_simple(bb1);

    // transform into two polynomials
    polynomial<double> pu1 =  tbc1.as_polynomial(point2d::u);
    polynomial<double> pv1 =  tbc1.as_polynomial(point2d::v);

    // get um and vm
    polynomial<double> um1 = bb1.max[point2d::u];
    polynomial<double> vm1 = bb1.max[point2d::v];

    // derive polynomials
    polynomial<double> dpu1 = derive(pu1);
    polynomial<double> dpv1 = derive(pv1);

    // generate target function
    polynomial<double> A1  = pu1 * (vm1 - pv1) + pv1 * (um1 - pu1);

    // derive target function
    polynomial<double> dA1 = derive(A1);

    // solve derivative of target function
    std::set<double> A1max = dA1.solve(bb1.min[point2d::u], bb1.max[point2d::u]);

    double tmax = 0.0;
    double Amax = 0.0;
    bool found = false;
    for ( std::set<double>::const_iterator i = A1max.begin(); i != A1max.end(); ++i )
    {
      if ( *i > 0.0 && *i < 1.0 )
      {
        double A = A1.evaluate(*i);
        if ( A > Amax )
        {
          tmax = *i;
          Amax = A;
          found = true;
        }
      }
    }

    CHECK ( found );
    CHECK ( A1.evaluate(tmax) > A1.evaluate(tmax + 0.001 * tmax) );
    CHECK ( A1.evaluate(tmax) > A1.evaluate(tmax - 0.001 * tmax) );
  }


}

