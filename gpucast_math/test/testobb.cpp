/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testobb.cpp
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
#include <cmath>

#include <gpucast/math/oriented_boundingbox.hpp>
#include <gpucast/math/parametric/pointmesh2d.hpp>
#include <gpucast/math/parametric/pointmesh3d.hpp>
#include <gpucast/math/oriented_boundingbox.hpp>
#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/matrix.hpp>
#include <boost/bind.hpp>

#include <gpucast/math/oriented_boundingbox_covariance_policy.hpp>
#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp>
#include <gpucast/math/oriented_boundingbox_random_policy.hpp>

using namespace gpucast::math;

namespace
{

  point3d rand_point3d(double min, double max)
  {
    double d = max - min;
    return point3d( min + d * double(rand())/RAND_MAX,
                    min + d * double(rand())/RAND_MAX,
                    min + d * double(rand())/RAND_MAX );
  }


  point2d rand_point2d(double min, double max)
  {
    double d = max - min;
    return point2d( min + d * double(rand())/RAND_MAX,
                    min + d * double(rand())/RAND_MAX );
  }

  gpucast::math::point3f min ( gpucast::math::point3f const& lhs, gpucast::math::point3f const& rhs)
  {
    return gpucast::math::point3f(std::min(lhs[0], rhs[0]),
                        std::min(lhs[1], rhs[1]),
                        std::min(lhs[2], rhs[2]),
                        std::min(lhs[3], rhs[3]));
  }

  gpucast::math::point3f max ( gpucast::math::point3f const& lhs, gpucast::math::point3f const& rhs)
  {
    return gpucast::math::point3f(std::max(lhs[0], rhs[0]),
                        std::max(lhs[1], rhs[1]),
                        std::max(lhs[2], rhs[2]),
                        std::max(lhs[3], rhs[3]));
  }

  gpucast::math::point3f 
  mult_mat4_float4 ( gpucast::math::point3f* O, gpucast::math::point3f const& p )
  {
    return gpucast::math::point3f ( O[0][0]*p[0] + O[0][1]*p[1] + O[0][2]*p[2] + O[0][3]*p[3], 
                          O[1][0]*p[0] + O[1][1]*p[1] + O[1][2]*p[2] + O[1][3]*p[3], 
                          O[2][0]*p[0] + O[2][1]*p[1] + O[2][2]*p[2] + O[2][3]*p[3], 
                          O[3][0]*p[0] + O[3][1]*p[1] + O[3][2]*p[2] + O[3][3]*p[3] );
  }

  bool bbox_intersect ( gpucast::math::point3f const& ray_origin,
                        gpucast::math::point3f const& ray_direction,
                        gpucast::math::point3f*       orientation,
                        gpucast::math::point3f*       orientation_inv,
                        gpucast::math::point3f const& low,
                        gpucast::math::point3f const& high,
                        gpucast::math::point3f const& center,
                        bool                backface_culling,
                        float&              t_min,
                        float&              t_max,
                        gpucast::math::point2f&       uv_min,
                        gpucast::math::point2f&       uv_max )
  {
    t_min = 0.0f; 
    t_max = 0.0f;

    gpucast::math::point3f local_origin     = ray_origin - center;
    local_origin[3]                = 1.0f;
    local_origin                  = mult_mat4_float4 ( orientation_inv, local_origin );
    gpucast::math::point3f local_direction  = mult_mat4_float4 ( orientation_inv, ray_direction );

    gpucast::math::point3f size            = high - low;

    gpucast::math::point3f tlow            = (low  - gpucast::math::point3f ( local_origin[0], local_origin[1], local_origin[2] )) / 
                                     gpucast::math::point3f ( local_direction[0], local_direction[1], local_direction[2] );
    gpucast::math::point3f thigh           = (high - gpucast::math::point3f ( local_origin[0], local_origin[1], local_origin[2] )) / 
                                     gpucast::math::point3f ( local_direction[0], local_direction[1], local_direction[2] );
  
    gpucast::math::point3f tmin            = min ( tlow, thigh );
    gpucast::math::point3f tmax            = max ( tlow, thigh );

    bool  min_intersect_found = false;

    // intersect with minimum planes
    gpucast::math::point3f xmin_intersect = local_origin + tmin[0] * local_direction;
    if ( xmin_intersect[1] >= low[1] && xmin_intersect[1] <= high[1] &&
         xmin_intersect[2] >= low[2] && xmin_intersect[2] <= high[2] )
    {
      gpucast::math::point3f local_hit    = local_origin + tmin[0] * local_direction;
      uv_min              = gpucast::math::point2f ( (local_hit[1] - low[1]) / size[1], (local_hit[2] - low[2]) / size[2] );
      t_min               = tmin[0];
      min_intersect_found = true;
    }

    gpucast::math::point3f ymin_intersect = local_origin + tmin[1] * local_direction;
    if ( ymin_intersect[0] >= low[0] && ymin_intersect[0] <= high[0] &&
         ymin_intersect[2] >= low[2] && ymin_intersect[2] <= high[2] )
    {
      gpucast::math::point3f local_hit    = local_origin + tmin[1] * local_direction;
      uv_min              = gpucast::math::point2f ( (local_hit[0] - low[0]) / size[0], (local_hit[2] - low[2]) / size[2] );
      t_min               = tmin[1];
      min_intersect_found = true;
    }

    gpucast::math::point3f zmin_intersect = local_origin + tmin[2] * local_direction;
    if ( zmin_intersect[0] >= low[0] && zmin_intersect[0] <= high[0] &&
         zmin_intersect[1] >= low[1] && zmin_intersect[1] <= high[1] )
    {
      gpucast::math::point3f local_hit    = local_origin + tmin[2] * local_direction;
      uv_min              = gpucast::math::point2f ( (local_hit[0] - low[0]) / size[0], (local_hit[1] - low[1]) / size[1] );
      t_min               = tmin[2];
      min_intersect_found = true;
    }
  
    // early exit if hit found 
    if ( backface_culling ) 
    {
      return min_intersect_found;
    }

    // intersect with maximum planes
    gpucast::math::point3f xmax_intersect = local_origin + tmax[0] * local_direction;
    if ( xmax_intersect[1] >= low[1] && xmax_intersect[1] <= high[1] &&
         xmax_intersect[2] >= low[2] && xmax_intersect[2] <= high[2] )
    {
      gpucast::math::point3f local_hit    = local_origin + tmax[0] * local_direction;
      uv_max              = gpucast::math::point2f ( (local_hit[1] - low[1]) / size[1], (local_hit[2] - low[2]) / size[2] );
      t_max               = tmax[0];
    }

    gpucast::math::point3f ymax_intersect = local_origin + tmax[1] * local_direction;
    if ( ymax_intersect[0] >= low[0] && ymax_intersect[0] <= high[0] &&
         ymax_intersect[2] >= low[2] && ymax_intersect[2] <= high[2] )
    {
      gpucast::math::point3f local_hit    = local_origin + tmax[1] * local_direction;
      uv_max              = gpucast::math::point2f ( (local_hit[0] - low[0]) / size[0], (local_hit[2] - low[2]) / size[2] );
      t_max               = tmax[1];
    }

    gpucast::math::point3f zmax_intersect = local_origin + tmax[2] * local_direction;
    if ( zmax_intersect[0] >= low[0] && zmax_intersect[0] <= high[0] &&
         zmax_intersect[1] >= low[1] && zmax_intersect[1] <= high[1] )
    {
      gpucast::math::point3f local_hit    = local_origin + tmax[2] * local_direction;
      uv_max              = gpucast::math::point2f ( (local_hit[0] - low[0]) / size[0], (local_hit[1] - low[1]) / size[1] );
      t_max               = tmax[2];
    }

    return min_intersect_found;// || max_intersect_found;
  }



}

SUITE (obb)
{

  TEST(ctor)
  {
    unsigned const NTESTS = 1;
    for ( unsigned i = 0; i != NTESTS; ++i )
    {
      std::vector<point3d> points ( 4 );
      std::generate ( points.begin(), points.end(), boost::bind(&rand_point3d, 5.0, 12.0) );

      oriented_boundingbox<point3d> bb(points.begin(), points.end(), covariance_policy<point3d>(-1000.0, 1000.0, 0.0000001) );

      std::vector<point3d> box_geometry;
      bb.generate_corners ( std::back_inserter(box_geometry) );
    }
  }

  TEST(bbox_intersect)
  {
    gpucast::math::matrix3f M = gpucast::math::make_rotation_x ( 0.0f ) * gpucast::math::make_rotation_y(0.0f) * gpucast::math::make_rotation_z(0.2f);
    gpucast::math::matrix3f Minv = M.inverse();
   
    gpucast::math::point3f orientation[4] = { M[0], M[1], M[2], gpucast::math::point3f(0.0f, 0.0f, 0.0f, 1.0f) };
    orientation[0][3] = 0.0f;
    orientation[1][3] = 0.0f;
    orientation[2][3] = 0.0f;
    gpucast::math::point3f orient_inv[4]  = { Minv[0], Minv[1], Minv[2], gpucast::math::point3f(0.0f, 0.0f, 0.0f, 1.0f) };
    orient_inv[0][3] = 0.0f;
    orient_inv[1][3] = 0.0f;
    orient_inv[2][3] = 0.0f;

    gpucast::math::point3f origin(3.0f, 1.0f, 0.0f, 1.0f);
    gpucast::math::point3f direction(2.0f, 1.0f, 0.0f, 0.0f);
    gpucast::math::point3f low(-2.0f, -2.0f, -2.0f);
    gpucast::math::point3f high(2.0f, 2.0f, 2.0f);
    gpucast::math::point3f center(7.0f, 2.0f, 0.0f, 1.0f);
    gpucast::math::point2f uvmin;
    gpucast::math::point2f uvmax;

    float tmin;
    float tmax;

    bbox_intersect(origin, direction, orientation, orient_inv, low, high, center, false, tmin, tmax, uvmin, uvmax);
  }


  TEST(test2d)
  {
    unsigned const NTESTS = 100;
    for ( unsigned i = 0; i != NTESTS; ++i )
    {
      std::vector<point2d> points ( 6 );
      std::generate ( points.begin(), points.end(), boost::bind(&rand_point2d, 3.0, 15.0) );
      //std::copy ( points.begin(), points.end(), std::ostream_iterator<point2d>(std::cout, "\n") );

      oriented_boundingbox<point2d> obb  (points.begin(), points.end(), covariance_policy<point2d>(-1000.0, 1000.0, 0.0000001) );
      axis_aligned_boundingbox<point2d>          aabb (points.begin(), points.end());

      pointmesh2d<point2d> mesh(points.begin(), points.end(), 2, 3);
      oriented_boundingbox<point2d> obb2 (mesh, partial_derivative_policy<point2d>() );

      //std::cout << obb2 << std::endl;
      //std::cout << obb.volume() << " : " << obb2.volume() << " : " << aabb.volume() << std::endl;
    }
  }


  TEST(overlap)
  {
    unsigned const NTESTS = 100;
    for ( unsigned i = 0; i != NTESTS; ++i )
    {
      // generate random coordinate system
      std::vector<point3d> dir1 ( 3 );
      std::vector<point3d> dir2 ( 3 );
      std::generate ( dir1.begin(), dir1.end(), boost::bind(&rand_point3d, -1.0, 1.0) );
      std::generate ( dir2.begin(), dir2.end(), boost::bind(&rand_point3d, -1.0, 1.0) );

      // normalize basis vectors
      dir1[0].normalize();
      dir1[1].normalize();
      dir1[2].normalize();

      dir2[0].normalize();
      dir2[1].normalize();
      dir2[2].normalize();

      matrix<double, 3, 3> m1;
      matrix<double, 3, 3> m2;

      m1[0][0] = dir1[0][0];
      m1[1][0] = dir1[0][1];
      m1[2][0] = dir1[0][2];

      m1[0][1] = dir1[1][0];
      m1[1][1] = dir1[1][1];
      m1[2][1] = dir1[1][2];

      m1[0][2] = dir1[2][0];
      m1[1][2] = dir1[2][1];
      m1[2][2] = dir1[2][2];

      m2[0][0] = dir2[0][0];
      m2[1][0] = dir2[0][1];
      m2[2][0] = dir2[0][2];

      m2[0][1] = dir2[1][0];
      m2[1][1] = dir2[1][1];
      m2[2][1] = dir2[1][2];

      m2[0][2] = dir2[2][0];
      m2[1][2] = dir2[2][1];
      m2[2][2] = dir2[2][2];

      // other parameters
      point3d center1 = rand_point3d(5.0, 7.0);
      point3d center2 = rand_point3d(6.0, 7.0);

      point3d high1   = rand_point3d(1.0, 4.0);
      point3d high2   = rand_point3d(1.0, 4.0);

      point3d low1    = -1.0 * high1;
      point3d low2    = -1.0 * high2;

      oriented_boundingbox<point3d> obb1 (m1, center1, low1, high1);
      oriented_boundingbox<point3d> obb2 (m2, center2, low2, high2);

      obb1.overlap(obb2);
      // not possible to check since it's random
    }
  }


  TEST(overlap2)
  {
    unsigned const NTESTS = 100;
    for ( unsigned i = 0; i != NTESTS; ++i )
    {

      double alpha = 2.0 * M_PI * (10.0/360.0);
      double beta  = 2.0 * M_PI * (40.0/360.0);

      // generate random coordinate system
      point3d a_dir1 (  std::cos(alpha), std::sin(alpha), 0.0 );
      point3d a_dir2 ( -std::sin(alpha), std::cos(alpha), 0.0 );

      point3d b_dir1 (  std::cos(beta),  std::sin(beta),  0.0 );
      point3d b_dir2 ( -std::sin(beta),  std::cos(beta),  0.0 );

      // normalize basis vectors
      a_dir1.normalize();
      a_dir2.normalize();
      b_dir1.normalize();
      b_dir2.normalize();

      matrix<double, 3, 3> m1;
      matrix<double, 3, 3> m2;

      m1[0][0] = a_dir1[0];
      m1[1][0] = a_dir1[1];
      m1[2][0] = a_dir1[2];

      m1[0][1] = a_dir2[0];
      m1[1][1] = a_dir2[1];
      m1[2][1] = a_dir2[2];

      m1[0][2] = 0.0;
      m1[1][2] = 0.0;
      m1[2][2] = 1.0;

      m2[0][0] = b_dir1[0];
      m2[1][0] = b_dir1[1];
      m2[2][0] = b_dir1[2];

      m2[0][1] = b_dir2[0];
      m2[1][1] = b_dir2[1];
      m2[2][1] = b_dir2[2];

      m2[0][2] = 0.0;
      m2[1][2] = 0.0;
      m2[2][2] = 1.0;

      // other parameters
      point3d center1 (3.0, 5.0, 0.0);
      point3d center2 (6.0, 7.0, 0.0);

      if ( i != 0 )
      {
        // result should be invariant to translations
        point3d rand_offset = rand_point3d(-20.0, 20.0);
        center1 += rand_offset;
        center2 += rand_offset;
      }

      point3d high1 (2.0, 4.0, 1.0);
      point3d high2 (1.0, 0.6, 1.0);
      point3d high3 (1.5, 0.6, 1.0);

      point3d low1    = -1.0 * high1;
      point3d low2    = -1.0 * high2;
      point3d low3    = -1.0 * high3;

      oriented_boundingbox<point3d> obb1 (m1, center1, low1, high1);
      oriented_boundingbox<point3d> obb2 (m2, center2, low2, high2);
      oriented_boundingbox<point3d> obb3 (m2, center2, low3, high3);

      bool k0 = obb1.overlap(obb2);
      bool k1 = obb1.overlap(obb3);
      bool k2 = obb2.overlap(obb1);
      bool k3 = obb3.overlap(obb1);

      CHECK(!k0);
      CHECK( k1);
      CHECK(!k2);
      CHECK( k3);
    }
  }


  TEST(overlap3)
  {
    unsigned const NTESTS = 100;
    for ( unsigned i = 0; i != NTESTS; ++i )
    {

      double alpha = 2.0 * M_PI * (10.0/360.0);

      // generate random coordinate system
      point3d dir1 (  std::cos(alpha), std::sin(alpha), 0.0 );
      point3d dir2 ( -std::sin(alpha), std::cos(alpha), 0.0 );

      // normalize basis vectors
      dir1.normalize();
      dir2.normalize();

      matrix<double, 3, 3> m;

      m[0][0] = dir1[0];
      m[1][0] = dir1[1];
      m[2][0] = dir1[2];

      m[0][1] = dir2[0];
      m[1][1] = dir2[1];
      m[2][1] = dir2[2];

      m[0][2] = 0.0;
      m[1][2] = 0.0;
      m[2][2] = 1.0;

      // other parameters
      point3d center (3.0, 5.0, 0.0);
      point3d high   (2.0, 4.0, 1.0);
      point3d low    (-1.0 * high);

      axis_aligned_boundingbox<point3d> aabb1 (point3d(1.5, 9.0, -1.0), point3d(5.0, 9.5,  1.1)); // overlaps
      axis_aligned_boundingbox<point3d> aabb2 (point3d(0.5, 9.0, -1.0), point3d(1.0, 10.0, 1.0)); // does not overlap
      axis_aligned_boundingbox<point3d> aabb3 (point3d(2.0, 4.0, -0.5), point3d(3.0, 6.0,  0.5)); // overlaps
      axis_aligned_boundingbox<point3d> aabb4 (point3d(2.0, 4.0,  2.0), point3d(3.0, 6.0,  3.0)); // does not overlap
      axis_aligned_boundingbox<point3d> aabb5 (point3d(5.0, 8.0, -0.5), point3d(5.5, 8.5,  0.5)); // does not overlap
      axis_aligned_boundingbox<point3d> aabb6 (point3d(0.0, 0.0, -0.5), point3d(6.0, 1.0,  2.5)); // overlap

      if ( i != 0 )
      {
        // result should be invariant to translations
        point3d rand_offset = rand_point3d(-20.0, 20.0);
        center   += rand_offset;
        aabb1.min += rand_offset;
        aabb1.max += rand_offset;
        aabb2.min += rand_offset;
        aabb2.max += rand_offset;
        aabb3.min += rand_offset;
        aabb3.max += rand_offset;
        aabb4.min += rand_offset;
        aabb4.max += rand_offset;
        aabb5.min += rand_offset;
        aabb5.max += rand_offset;
        aabb6.min += rand_offset;
        aabb6.max += rand_offset;
      }

      oriented_boundingbox<point3d>     obb   (m, center, low, high);

      oriented_boundingbox<point3d>     obb1  (aabb1);
      oriented_boundingbox<point3d>     obb2  (aabb2);
      oriented_boundingbox<point3d>     obb3  (aabb3);
      oriented_boundingbox<point3d>     obb4  (aabb4);
      oriented_boundingbox<point3d>     obb5  (aabb5);
      oriented_boundingbox<point3d>     obb6  (aabb6);

      bool k1 = obb.overlap(aabb1);
      bool k2 = obb.overlap(aabb2);
      bool k3 = obb.overlap(aabb3);
      bool k4 = obb.overlap(aabb4);
      bool k5 = obb.overlap(aabb5);
      bool k6 = obb.overlap(aabb6);

      // vice versa should result in same result
      bool k7 = obb1.overlap(obb);
      bool k8 = obb2.overlap(obb);
      bool k9 = obb3.overlap(obb);
      bool k10= obb4.overlap(obb);
      bool k11= obb5.overlap(obb);
      bool k12= obb6.overlap(obb);

      CHECK( k1);
      CHECK(!k2);
      CHECK( k3);
      CHECK(!k4);
      CHECK(!k5);
      CHECK( k6);

      CHECK(k1 == k7);
      CHECK(k2 == k8);
      CHECK(k3 == k9);
      CHECK(k4 == k10);
      CHECK(k5 == k11);
      CHECK(k6 == k12);
    }
  }

   TEST(is_inside_point)
  {
    double alpha = 2.0 * M_PI * (10.0/360.0);

    // generate random coordinate system
    point3d dir1 (  std::cos(alpha), std::sin(alpha), 0.0 );
    point3d dir2 ( -std::sin(alpha), std::cos(alpha), 0.0 );

    // normalize basis vectors
    dir1.normalize();
    dir2.normalize();

    matrix<double, 3, 3> m;

    m[0][0] = dir1[0];
    m[1][0] = dir1[1];
    m[2][0] = dir1[2];

    m[0][1] = dir2[0];
    m[1][1] = dir2[1];
    m[2][1] = dir2[2];

    m[0][2] = 0.0;
    m[1][2] = 0.0;
    m[2][2] = 1.0;

    // other parameters
    point3d center (3.0, 5.0, 0.0);
    point3d high   (2.0, 4.0, 1.0);
    point3d low    (-1.0 * high);

    oriented_boundingbox<point3d>     obb   (m, center, low, high);

    CHECK( obb.is_inside(point3d(5.0, 5.0,  0.0)));
    CHECK( obb.is_inside(point3d(4.0, 9.0,  0.9)));
    CHECK( obb.is_inside(point3d(2.0, 1.0, -0.9)));
    CHECK( obb.is_inside(point3d(5.0, 1.5,  0.0)));
    CHECK( obb.is_inside(point3d(1.0, 8.5,  0.0)));

    CHECK( !obb.is_inside(point3d(5.0, 5.0, 2.0)));
    CHECK( !obb.is_inside(point3d(5.0, 5.0,-2.0)));
    CHECK( !obb.is_inside(point3d(5.0, 1.0, 0.0)));
    CHECK( !obb.is_inside(point3d(5.0, 6.0, 0.0)));
    CHECK( !obb.is_inside(point3d(3.0, 9.5, 0.0)));
    CHECK( !obb.is_inside(point3d(1.0, 1.0, 0.0)));
  }



  TEST(volume2d)
  {
    matrix<double, 2, 2> orientation;

    point2d high ( 3.0, 2.0);
    point2d low  (-1.0, -3.0);

    orientation[0][0] = 1.0;
    orientation[1][0] = 0.0;
    orientation[0][1] = 4.0/5.0;
    orientation[1][1] = 3.0/5.0;

    oriented_boundingbox<point2d> obb(orientation, point2d(0.0, 0.0), low, high);
    CHECK_CLOSE( obb.volume(), 12.0, 0.000001 );

    matrix<double, 3, 3> orientation3d;

    point3d high3d ( 3, 2, 1);
    point3d low3d  ( -1, -3, 0);

    orientation3d[0][0] = 1.0;
    orientation3d[1][0] = 0.0;
    orientation3d[2][0] = 0.0;

    orientation3d[0][1] = 4.0/5.0;
    orientation3d[1][1] = 3.0/5.0;
    orientation3d[2][1] = 0.0;

    orientation3d[0][2] = 0.0;
    orientation3d[1][2] = 0.0;
    orientation3d[2][2] = 1.0;

    oriented_boundingbox<point3d> obb3d(orientation3d, point3d(0.0, 0.0, 0.0), low3d, high3d);
    CHECK_CLOSE( obb3d.volume(), 12.0, 0.000001 );
  }


}

