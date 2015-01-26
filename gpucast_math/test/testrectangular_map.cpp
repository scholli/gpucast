/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testrectangular_map.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <vector>
#include <iostream>

#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_binary.hpp>

using namespace gpucast::math;
using namespace gpucast::math::domain;

SUITE (polynomial)
{

  TEST(creation)
  {
    beziercurve2d bc1, bc2, bc3, bc4, bc5, bc6;

    // first loop
    bc1.add(point2d(0,  0));
    bc1.add(point2d(10, 0));
    bc1.add(point2d(10, 6));

    bc2.add(point2d(10, 6));
    bc2.add(point2d(11, 8));
    bc2.add(point2d(3,  9));

    bc3.add(point2d(3, 9));
    bc3.add(point2d(0, 7));
    bc3.add(point2d(0, 0));

    std::vector<domain::contour<double>::curve_ptr> loop1;                                          
    loop1.push_back ( contour<double>::curve_ptr ( new contour<double>::curve_type(bc1) ) );
    loop1.push_back ( contour<double>::curve_ptr ( new contour<double>::curve_type(bc2) ) );
    loop1.push_back ( contour<double>::curve_ptr ( new contour<double>::curve_type(bc3) ) );

    contour<double> contour1 (loop1.begin(), loop1.end());

    // second loop
    bc4.add(point2d(2, 2));
    bc4.add(point2d(3, 4));
    bc4.add(point2d(2, 5));

    bc5.add(point2d(2, 5));
    bc5.add(point2d(1, 4));
    bc5.add(point2d(2, 2));

    std::vector<contour<double>::curve_ptr> loop2;
    loop2.push_back ( contour<double>::curve_ptr ( new contour<double>::curve_type(bc4) ) );
    loop2.push_back ( contour<double>::curve_ptr ( new contour<double>::curve_type(bc5) ) );
    contour<double> contour2 ( loop2.begin(), loop2.end() );

    contour_map_binary<double> cmb;

    cmb.add(contour1);
    cmb.add(contour2);

    cmb.initialize();
  }

}



