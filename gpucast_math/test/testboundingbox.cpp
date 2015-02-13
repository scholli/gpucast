/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/boundingbox.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <boost/bind.hpp>

using namespace gpucast::math;

SUITE (boundingbox_classes)
{
  TEST(default_ctor)
  {
    point3d p0(1.0, 4.0, 2.0);
    point3d p1(4.0, 5.0, 3.0);

    axis_aligned_boundingbox<point3d> b(p0, p1);

    CHECK( p0 == b.min );
    CHECK( p1 == b.max );
  }

  TEST(swap)
  {
    point3d p0(1.0, 4.0, 2.0);
    point3d p1(4.0, 5.0, 3.0);

    point3d p2(2.0, 7.0, 3.0);
    point3d p3(3.0, 9.0, 6.0);

    axis_aligned_boundingbox<point3d> b0(p0, p1);
    axis_aligned_boundingbox<point3d> b1(p2, p3);

    b0.swap(b1);

    CHECK( p2 == b0.min );
    CHECK( p3 == b0.max );

    CHECK( p0 == b1.min );
    CHECK( p1 == b1.max );
  }

  TEST(merge)
  {
    point3d p0(-1.0, 4.0, -2.0);
    point3d p1(4.0, 5.0, 3.0);

    point3d p2(2.0, 7.0, -3.0);
    point3d p3(3.0, 9.0, 6.0);

    axis_aligned_boundingbox<point3d> b0(p0, p1);
    axis_aligned_boundingbox<point3d> b1(p2, p3);

    axis_aligned_boundingbox<point3d> b3 = merge(b0, b1);

    CHECK (b3.min == point3d(-1.0, 4.0, -3.0));
    CHECK (b3.max == point3d( 4.0, 9.0,  6.0));
  }

  TEST(center)
  {
    point3d p0(-1.0, 4.0, -2.0);
    point3d p1(4.0, 3.0, 3.0);
    point3d p2(5.0, 6.0, 5.0);

    axis_aligned_boundingbox<point3d> b0(p0, p1);
    axis_aligned_boundingbox<point3d> b1(p0, p2);

    CHECK_CLOSE (b0.center().distance(point3d(1.5, 3.5, 0.5)), 0.0, 0.000001);
    CHECK_CLOSE (b1.center().distance(point3d(2.0, 5.0, 1.5)), 0.0, 0.000001);
  }


  TEST(valid)
  {
    point3d p0(-1.0, 4.0, -2.0);
    point3d p1(4.0, 3.0, 3.0);
    point3d p2(4.0, 6.0, 3.0);

    axis_aligned_boundingbox<point3d> b0(p0, p1);
    axis_aligned_boundingbox<point3d> b1(p0, p2);

    CHECK (!b0.valid());
    CHECK (b1.valid());
  }

  TEST(inside)
  {
    point3d p0(-1.0, 4.0, -2.0);
    point3d p1(4.0, 6.0, 3.0);

    point3d p2(4.0, 6.1, 3.0); // should not
    point3d p3(4.0, 5.0, 2.5); // should 
    point3d p4(0.0, 0.0, 0.0); // should not
    point3d p5(0.0, 6.0, 0.0); // should 
    point3d p6(-1.0, 6.0, -2.0); // should 
    point3d p7(-1.01, 5.0, 0.0); // should not
    point3d p8(2.0, 3.99, -1.9); // should not

    axis_aligned_boundingbox<point3d> b0(p0, p1);

    CHECK(b0.is_inside(p0));
    CHECK(b0.is_inside(p1));
    CHECK(!b0.is_inside(p2));
    CHECK(b0.is_inside(p3));
    CHECK(!b0.is_inside(p4));
    CHECK(b0.is_inside(p5));
    CHECK(b0.is_inside(p6));
    CHECK(!b0.is_inside(p7));
    CHECK(!b0.is_inside(p8));
  }

  
  TEST(overlap)
  {
    point2d p0(-1.0, 4.0);
    point2d p1( 4.0, 6.0);

    point2d p2( 1.0, 2.0);
    point2d p3( 5.0, 7.0);

    axis_aligned_boundingbox<point2d> b0(p0, p1);
    axis_aligned_boundingbox<point2d> b1(p2, p3);

    std::vector<point2d> corners0;
    std::vector<point2d> corners1;

    b0.generate_corners(std::back_inserter(corners0));
    b1.generate_corners(std::back_inserter(corners1));

    CHECK(b0.overlap(b1));
    CHECK(b1.overlap(b0));
  }

  TEST(generate_corners)
  {
    point3d p0(-1.0, 4.0, 2.0);
    point3d p1( 4.0, 6.0, 3.0);

    axis_aligned_boundingbox<point3d> b(p0, p1);
    std::vector<point3d> corners;
    b.generate_corners ( std::back_inserter(corners) );

    CHECK (corners.size() == 8);
    CHECK (corners[0] == point3d(-1.0, 4.0, 2.0));
    CHECK (corners[1] == point3d( 4.0, 4.0, 2.0));
    CHECK (corners[2] == point3d(-1.0, 6.0, 2.0));
    CHECK (corners[3] == point3d( 4.0, 6.0, 2.0));
    CHECK (corners[4] == point3d(-1.0, 4.0, 3.0));
    CHECK (corners[5] == point3d( 4.0, 4.0, 3.0));
    CHECK (corners[6] == point3d(-1.0, 6.0, 3.0));
    CHECK (corners[7] == point3d( 4.0, 6.0, 3.0));
  }


  TEST(volume)
  {
    point3d p0(-1.0, 4.0, 2.0);
    point3d p1( 4.0, 6.0, 3.0);

    axis_aligned_boundingbox<point3d> b(p0, p1);

    CHECK_CLOSE(b.volume(), 10.0, 0.0000001);
  }

  TEST(generate)
  {
    std::vector<point3d> points(5);
    points[0] = point3d(-1.0, 4.0, 2.0);
    points[1] = point3d( 1.0, 1.0,-2.0);
    points[2] = point3d(-5.0, 2.0,-1.0);
    points[3] = point3d(-3.0, 5.0, 0.0);
    points[4] = point3d( 6.0, 6.0, 3.0);
    
    axis_aligned_boundingbox<point3d> b(points.begin(), points.end());

    CHECK (b.min == point3d(-5.0, 1.0, -2.0) );
    CHECK (b.max == point3d( 6.0, 6.0,  3.0) );
  }

  TEST(split)
  {
    point3d p0 (-1.0, -4.0, -2.0);
    point3d p1 ( 3.0,  2.0, 4.0);

    point2d p2 ( 1.0, -2.0);
    point2d p3 ( 3.0,  2.0);
    
    axis_aligned_boundingbox<point3d> b0(p0, p1);
    axis_aligned_boundingbox<point2d> b1(p2, p3);

    std::vector<axis_aligned_boundingbox<point3d>::pointer_type> split3d;
    std::vector<axis_aligned_boundingbox<point2d>::pointer_type> split2d;

    b0.uniform_split_ptr(std::inserter(split3d, split3d.end()));
    b1.uniform_split_ptr(std::inserter(split2d, split2d.end()));

    CHECK (split3d.size() == 8);
    CHECK (split2d.size() == 4);

    //std::copy(split2d.begin(), split2d.end(), std::ostream_iterator<axis_aligned_boundingbox<point2d> > (std::cout, "\n"));
    std::for_each(split2d.begin(), split2d.end(), boost::bind(&axis_aligned_boundingbox<point2d>::print, _1, boost::ref(std::cout)));

    CHECK (split3d[0]->min == point3d(-1.0, -4.0, -2.0) );
    CHECK (split3d[0]->max == point3d( 1.0, -1.0,  1.0) );

    CHECK (split3d[1]->min == point3d( 1.0, -4.0, -2.0) );
    CHECK (split3d[1]->max == point3d( 3.0, -1.0,  1.0) );
                    
    CHECK (split3d[2]->min == point3d(-1.0, -1.0, -2.0) );
    CHECK (split3d[2]->max == point3d( 1.0,  2.0,  1.0) );
                    
    CHECK (split3d[3]->min == point3d( 1.0, -1.0, -2.0) );
    CHECK (split3d[3]->max == point3d( 3.0,  2.0,  1.0) );
                   
    CHECK (split3d[4]->min == point3d(-1.0, -4.0, 1.0) );
    CHECK (split3d[4]->max == point3d( 1.0, -1.0, 4.0) );
 
    CHECK (split3d[5]->min == point3d( 1.0, -4.0, 1.0) );
    CHECK (split3d[5]->max == point3d( 3.0, -1.0, 4.0) );

    CHECK (split3d[6]->min == point3d(-1.0, -1.0, 1.0) );
    CHECK (split3d[6]->max == point3d( 1.0,  2.0, 4.0) );
     
    CHECK (split3d[7]->min == point3d( 1.0, -1.0, 1.0) );
    CHECK (split3d[7]->max == point3d( 3.0,  2.0, 4.0) );

    CHECK (split2d[0]->min == point2d(1.0, -2.0));
    CHECK (split2d[0]->max == point2d(2.0,  0.0));

    CHECK (split2d[1]->min == point2d(2.0, -2.0));
    CHECK (split2d[1]->max == point2d(3.0,  0.0));

    CHECK (split2d[2]->min == point2d(1.0, 0.0));
    CHECK (split2d[2]->max == point2d(2.0, 2.0));

    CHECK (split2d[3]->min == point2d(2.0, 0.0));
    CHECK (split2d[3]->max == point2d(3.0, 2.0));
  }

  TEST(distance)
  {
    point2d p0(-1.0, 4.0);
    point2d p1(5.0, 7.0);

    axis_aligned_boundingbox<point2d> b(p0, p1);

    point2d p2(-2.0, 2.0); // 
    point2d p3(-2.0, 5.0); // 
    point2d p4(-2.0, 9.0); // 

    point2d p5(2.0, 2.0); // 
    point2d p6(2.0, 5.0); // 
    point2d p7(2.0, 9.0); // 

    point2d p8(7.0, 2.0); // 
    point2d p9(7.0, 5.0); // 
    point2d p10(7.0, 9.0); // 

    CHECK_CLOSE(b.distance(p2), point2d(1, 2, 1).abs(), 0.000001);
    CHECK_CLOSE(b.distance(p3), point2d(1, 0, 1).abs(), 0.000001);
    CHECK_CLOSE(b.distance(p4), point2d(1, 2, 1).abs(), 0.000001);

    CHECK_CLOSE(b.distance(p5), point2d(0, 2, 1).abs(), 0.000001);
    CHECK_CLOSE(b.distance(p6), point2d(0, 0, 1).abs(), 0.000001);
    CHECK_CLOSE(b.distance(p7), point2d(0, 2, 1).abs(), 0.000001);

    CHECK_CLOSE(b.distance(p8), point2d(2, 2, 1).abs(), 0.000001);
    CHECK_CLOSE(b.distance(p9), point2d(2, 0, 1).abs(), 0.000001);
    CHECK_CLOSE(b.distance(p10), point2d(2, 2, 1).abs(), 0.000001);
  }

}
