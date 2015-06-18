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

#include <unittest++/UnitTest++.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <gpucast/math/util/prefilter3d.hpp>
#include <gpucast/math/util/prefilter2d.hpp>
#include <gpucast/math/parametric/point.hpp>

using namespace gpucast::math;
using namespace gpucast::math::util;

SUITE (functions_pixel_coverage)
{
  TEST(alpha_to_equation)
  {
    prefilter3d<point3d> pre3d;

    for (double angle = 0.0; angle <= 2 * M_PI; angle += M_PI/8.0) 
    {
      for (double d1 = -0.5; d1 <= 0.5; d1 += 0.1) 
      {
        for (double d2 = -0.5; d2 <= 0.5; d2 += 0.1) 
        {
          //std::cout << angle << " : " << d1 << " : " << d2 << std::endl;
          //std::cout << "coverage : " << pre3d(point3d(angle, d1, d2)) << std::endl;
        }
      }
    }
  }

  TEST(polar_to_euclid)
  {
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());
    int const MAX_TESTS = 100;

    for (int i = 0; i != MAX_TESTS; ++i) {

      float x = 2.0*(std::rand() / RAND_MAX) - 1.0;
      float y = 2.0*(std::rand() / RAND_MAX) - 1.0;

      point2f euclid(x, y);
      
      euclid_to_polar<point2f> conv;
      polar_to_euclid<point2f> backconv;

      point2f polar = conv(euclid);
      point2f euclid2 = backconv(polar);

      CHECK_CLOSE(euclid2[0], euclid[0], 0.0001);
      CHECK_CLOSE(euclid2[1], euclid[1], 0.0001);
    }
  }
  
}
