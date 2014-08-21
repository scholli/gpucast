/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testdiscretizer3d.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

#include <gpucast/math/util/discretizer3d.hpp>
#include <gpucast/math/util/prefilter3d.hpp>

using namespace gpucast::math;
using namespace gpucast::math::util;

SUITE (discretizer3d  )
{

  TEST(test_discretization)
  {
    discretizer3d<point3f> d3d(4, 3, 5, 1.0f, 2.0f, 4.0f, 7.0f, -2.0f, -1.0f);
    std::vector<point3f> discrete(4*3*5);

    std::generate(discrete.begin(), discrete.end(), d3d);
    CHECK_EQUAL(discrete[4], point3f(1.0f, 5.5f, -2.0f));
    CHECK_EQUAL(discrete[59], point3f(2.0f, 7.0f, -1.0f));
  }

  TEST(test_prefilter3d)
  {
    prefilter3d<point3f> prefilter;
    point3f f0 = prefilter(point3f(0.0, 0.4f, -0.5f));
    //std::cout << f0 << std::endl;
  }
  
}
