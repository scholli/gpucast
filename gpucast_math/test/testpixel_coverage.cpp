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

#include <cmath>
#include <iostream>
#include <gpucast/math/util/prefilter3d.hpp>
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
  
}
