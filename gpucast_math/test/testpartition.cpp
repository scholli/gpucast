/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testpartition.cpp
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

#include <gpucast/math/parametric/domain/partition/double_binary/partition.hpp>


using namespace gpucast::math;
using namespace gpucast::math::domain;

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

SUITE (partition_class)
{

  TEST(add)
  {
    std::shared_ptr<beziercurve2d> bc(new beziercurve2d(create_circle()));

    std::vector<std::shared_ptr<beziercurve2d>> curve_loop;
    curve_loop.push_back(bc);

    partition<point2d> part(curve_loop.begin(), curve_loop.end());
  }

  TEST(split_and_partition)
  {
	  point2d pA(0.5,  0.0);
    point2d pB(1.0,  0.5);
    point2d pC(0, 1.0);

    std::shared_ptr<beziercurve2d> ab(new beziercurve2d());
    std::shared_ptr<beziercurve2d> bc(new beziercurve2d());
    std::shared_ptr<beziercurve2d> ca(new beziercurve2d());
    std::shared_ptr<beziercurve2d> acba(new beziercurve2d());
	  //std::shared_ptr<beziercurve2d> circle(new beziercurve2d(create_circle()));

	  ab->add(pA);
	  ab->add(pB);

	  bc->add(pB);
	  bc->add(pC);

	  ca->add(pC);
	  ca->add(pA);

	  acba->add(pA);
	  acba->add(pC);
	  acba->add(pB);
	  acba->add(pA);
	
	  std::set<double> extremasu;
	  acba->extrema(point2d::u, extremasu, 64);	

	  std::set<double> extremasv;
	  acba->extrema(point2d::v, extremasv, 64);	

    std::vector<std::shared_ptr<beziercurve2d>> curvelist;
	  //curvelist.push_back(ab);
	  //curvelist.push_back(bc);
	  //curvelist.push_back(ca);
	  curvelist.push_back(acba);
	  //curvelist.push_back(circle);

	  partition<beziercurve2d::point_type> p(curvelist.begin(), curvelist.end());
	  p.initialize();
	  // p.print(std::cout);



	  partition<beziercurve2d::point_type>::const_iterator viter = p.begin();
	  partition<beziercurve2d::point_type>::const_iterator hiter;


	  for (; viter != p.end(); ++viter) {
		  //(*viter)->print(std::cout, "END\n");
		  // (*viter)->get_horizontal_interval();
		  //(*viter)->get_vertical_interval();
		  std::cout << "vertical partition, v : " << (*viter)->get_vertical_interval() << std::endl;
		  for(partition<beziercurve2d::point_type>::cell_ptr_set::const_iterator cell = (*viter)->begin(); cell != (*viter)->end(); ++cell)
		  {
			  std::cout << "cell, u : " << (*cell)->get_horizontal_interval() << std::endl;
			  std::cout << "intersections : " << (*cell)->intersections() << std::endl;
			  std::cout << "curves to intersect : " << (*cell)->size() << std::endl;
		  }

	  }

	  //std::shared_ptr<interval<beziercurve2d::point_type::value_type>> i(new
	  //	interval<beziercurve2d::point_type::value_type>(p.get_vertical_interval()));

  }



  TEST(split_and_partition2)
  {
    std::shared_ptr<gpucast::math::beziercurve2d> bc(new gpucast::math::beziercurve2d);

    gpucast::math::point2d p0(-0.8, 0.7);
    gpucast::math::point2d p1(-0.4, -0.8);
    gpucast::math::point2d p2(-0.2, 0.1);
    gpucast::math::point2d p3(0.7, 0.7);
    gpucast::math::point2d p4(0.9, -0.6);

    bc->add(p0);
    bc->add(p1);
    bc->add(p2);
    bc->add(p3);
    bc->add(p4);

    std::vector<std::shared_ptr<gpucast::math::beziercurve2d>> curves;
    curves.push_back(bc);

    partition<beziercurve2d::point_type> p(curves.begin(), curves.end());
    p.initialize();
  }
}