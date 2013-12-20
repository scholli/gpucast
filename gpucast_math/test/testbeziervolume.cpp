/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testbeziervolume.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#if WIN32
  #include <UnitTest++.h>
#else
  #include <unittest++/UnitTest++.h>
#endif

#include <gpucast/math/parametric/pointmesh3d.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/beziersurface.hpp>

#include <boost/bind.hpp>
#include <algorithm>


using namespace gpucast::math;

SUITE (beziervolume_class)
{
  TEST(check_evaluation)
  {
    srand(7);

    std::size_t width  = 3;
    std::size_t height = 3;
    std::size_t depth  = 3;

    pointmesh3d<point3d> mesh;

    for (std::size_t z = 0; z != depth; ++z)
    {
      for (std::size_t y = 0; y != height; ++y)
      {
        for (std::size_t x = 0; x != width; ++x)
        {
          double distort_max    = 0.0;
          double distort_x      = distort_max * 2.0 * (double(rand())/RAND_MAX - 0.5);
          double distort_y      = distort_max * 2.0 * (double(rand())/RAND_MAX - 0.5);
          double distort_z      = distort_max * 2.0 * (double(rand())/RAND_MAX - 0.5);

          point3d p = point3d(double(x) + distort_x, 
                              double(y) + distort_y, 
                              double(z) + distort_z);

          mesh.push_back(p);
        }
      }
    }

    mesh.width (3);
    mesh.height(3);
    mesh.depth (3);

    beziervolume<point3d> bv(mesh);

    point3d p0;
    bv.evaluate(0.3, 0.4, 0.2, p0);

    point3d p1, dv1, du1, dw1;
    point3d p2, dv2, du2, dw2;
    bv.evaluate(0.3, 0.4, 0.2, p1, du1, dv1, dw1);
    bv.evaluate(0.3, 0.4, 0.2, p2, du2, dv2, dw2, horner<point3d>());

    CHECK_CLOSE(p0.distance(p1), 0.0, 0.000001);
    CHECK_CLOSE(p1.distance(p2), 0.0, 0.000001);
    CHECK_CLOSE(du1.distance(du2), 0.0, 0.000001);
    CHECK_CLOSE(dv1.distance(dv2), 0.0, 0.000001);
    CHECK_CLOSE(dw1.distance(dw2), 0.0, 0.000001);
  }

  TEST(slice)
  {
    std::size_t width  = 4;
    std::size_t height = 6;
    std::size_t depth  = 5;

    pointmesh3d<point3d> mesh;

    for (std::size_t z = 0; z != depth; ++z)
    {
      for (std::size_t y = 0; y != height; ++y)
      {
        for (std::size_t x = 0; x != width; ++x)
        {
          point3d p = point3d(double(x), 
                              double(y), 
                              double(z));

          mesh.push_back(p);
        }
      }
    }

    mesh.width (width);
    mesh.height(height);
    mesh.depth (depth);

    beziervolume<point3d> bv(mesh);

    beziersurface<point3d> bs = bv.slice(2, 2);
  }

  TEST(ocsplit)
  {
    std::size_t width  = 4;
    std::size_t height = 6;
    std::size_t depth  = 5;

    pointmesh3d<point3d> mesh;

    mesh.width (width);
    mesh.height(height);
    mesh.depth (depth);

    for (std::size_t z = 0; z != depth; ++z)
    {
      for (std::size_t y = 0; y != height; ++y)
      {
        for (std::size_t x = 0; x != width; ++x)
        {
          point3d p = point3d(double(x), 
                              double(y), 
                              double(z));

          mesh.push_back(p);
        }
      }
    }

   beziervolume<point3d> bv(mesh);

   beziervolume<point3d>::array_type split = bv.ocsplit();
  
   for (std::size_t u = 0; u != 2; ++u) {
     for (std::size_t v = 0; v != 2; ++v) {
       for (std::size_t w = 0; w != 2; ++w) {
         //std::cout << split[u][v][w].bbox() << std::endl;
       }
     }
   }
  }
}
