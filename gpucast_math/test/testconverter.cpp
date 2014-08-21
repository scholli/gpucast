/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testconverter.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <list>
#include <cstdio>

#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/parametric/nurbscurve.hpp>
#include <gpucast/math/parametric/nurbssurface.hpp>
#include <gpucast/math/parametric/nurbsvolume.hpp>
#include <gpucast/math/parametric/algorithm/converter.hpp>


using namespace gpucast::math;

SUITE (class_converter)
{

  float random (float min, float max)
  {
    float rel_value = float(rand())/RAND_MAX;
    return min + rel_value * fabs(max - min);
  }


  nurbscurve2d create_testcurve()
  {
    nurbscurve2d nc;

    nc.add(point2d( 1, 3, 2));
    nc.add(point2d(-3, 2, 4));
    nc.add(point2d(-4, 5, 1));
    nc.add(point2d( 0, 8, 4));
    nc.add(point2d( 5, 4, 2));
    nc.add(point2d( 1, 2, 3));
    nc.add(point2d( 3, 0, 4));
    nc.add(point2d( 6, 2, 2));

    std::vector<double> knots;

    nc.add_knot(0);
    nc.add_knot(0);
    nc.add_knot(0);
    nc.add_knot(0);
    nc.add_knot(1);
    nc.add_knot(3);
    nc.add_knot(3);
    nc.add_knot(4);
    nc.add_knot(6);
    nc.add_knot(6);
    nc.add_knot(6);
    nc.add_knot(6);

    nc.degree(3);
    return nc;
  }


  nurbsvolume<point3d> create_volume()
  {
    nurbsvolume<point3d> nv;

    std::multiset<float> knots_u, knots_v, knots_w;

    std::size_t order_u = 4;
    std::size_t order_v = 5;
    std::size_t order_w = 3;

    nv.degree_u (order_u - 1);
    nv.degree_v (order_v - 1);
    nv.degree_w (order_w - 1);

    for (unsigned int u = 0; u != order_u; ++u)
    {
      knots_u.insert(0.f);
      knots_u.insert(1.f);
    }
    knots_u.insert(random(0.1f, 0.9f));
    knots_u.insert(random(0.1f, 0.9f));

    for (unsigned int v = 0; v != order_v; ++v)
    {
      knots_v.insert(0.0f);
      knots_v.insert(1.0f);
    }
    knots_v.insert(random(0.1f, 0.9f));
    knots_v.insert(random(0.1f, 0.9f));
    knots_v.insert(random(0.1f, 0.9f));

    for (unsigned int w = 0; w != order_w; ++w)
    {
      knots_w.insert(0.0f);
      knots_w.insert(1.0f);
    }
    knots_w.insert(random(0.1f, 0.9f));

    nv.knotvector_u(knots_u.begin(), knots_u.end());
    nv.knotvector_v(knots_v.begin(), knots_v.end());
    nv.knotvector_w(knots_w.begin(), knots_w.end());

    nv.resize(knots_u.size() - order_u,
              knots_v.size() - order_v,
              knots_w.size() - order_w);

    for (unsigned int w = 0; w != knots_w.size() - order_w; ++w)
    {
      for (unsigned int v = 0; v != knots_v.size() - order_v; ++v)
      {
        for (unsigned int u = 0; u != knots_u.size() - order_u; ++u)
        {
          point3d p(u + random(-0.5f, 0.5f), v + random(-0.5f, 0.5f), w + random(-0.5f, 0.5f));
          nv.set_point(u,v,w,p);
        }
      }
    }

    nv.numberofpoints_u(knots_u.size() - order_u);
    nv.numberofpoints_v(knots_v.size() - order_v);
    nv.numberofpoints_w(knots_w.size() - order_w);

    return nv;
  }






  TEST(convert_surface)
  {
    std::vector<point3d> pts(20);
    pts[0]  = point3d(0.0, 0.0, 0.3);
    pts[1]  = point3d(1.6, 0.0, 1.0);
    pts[2]  = point3d(2.4, 0.0, 2.0);
    pts[3]  = point3d(3.0, 0.0, 0.1);

    pts[4]  = point3d(0.0, 1.3, 0.4);
    pts[5]  = point3d(1.0, 1.2, 1.0);
    pts[6]  = point3d(2.0, 1.0, 1.0);
    pts[7]  = point3d(3.1, 1.1, 0.5);

    pts[8]  = point3d(0.0, 2.0, 0.1);
    pts[9]  = point3d(1.1, 2.2, 1.0);
    pts[10] = point3d(2.3, 2.3, 1.0);
    pts[11] = point3d(3.3, 2.4, 0.1);

    pts[12] = point3d(0.0, 3.0, 0.6);
    pts[13] = point3d(1.1, 3.2, 1.3);
    pts[14] = point3d(2.5, 3.5, 2.1);
    pts[15] = point3d(3.1, 3.0, 0.5);

    pts[16] = point3d(0.0, 4.3, 1.3);
    pts[17] = point3d(1.1, 4.0, 1.1);
    pts[18] = point3d(2.2, 4.1, 1.2);
    pts[19] = point3d(3.0, 4.0, 1.3);

    nurbssurface3d ns;
    ns.set_points(pts.begin(), pts.end());

    ns.numberofpoints_u(4);
    ns.numberofpoints_v(5);

    ns.degree_u(2);
    ns.degree_v(2);

    std::vector<double> ku(7);
    ku[0] = 0.0;
    ku[1] = 0.0;
    ku[2] = 0.0;
    ku[3] = 0.3;
    ku[4] = 1.0;
    ku[5] = 1.0;
    ku[6] = 1.0;

    std::vector<double> kv(8);
    kv[0] = 0.0;
    kv[1] = 0.0;
    kv[2] = 0.0;
    kv[3] = 0.3;
    kv[4] = 0.4;
    kv[5] = 1.0;
    kv[6] = 1.0;
    kv[7] = 1.0;

    ns.knotvector_u(ku.begin(), ku.end());
    ns.knotvector_v(kv.begin(), kv.end());

    converter<point3d> conv3d;
    std::list<beziersurface_from_nurbs<point3d> > splits;
    conv3d.convert(ns, std::back_inserter(splits));

    CHECK(splits.size() == 6);
  }

  TEST(convert_curve)
  {
    nurbscurve2d nc = create_testcurve();
    converter<point2d> conv2d;

    std::list<beziercurve2d> bcs;
    conv2d.convert(nc, std::back_inserter(bcs));

    CHECK(bcs.size() == 4);
  }



  TEST(convert_volume)
  {
    nurbsvolume<point3d> nv = create_volume();
    converter<point3d> converter;

    std::vector<beziervolume<point3d> > bezier_volumes;
    std::vector<gpucast::math::beziervolumeindex> indices;
    converter.convert(nv, std::back_inserter(bezier_volumes), indices);
  }
}

