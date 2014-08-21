/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testnurbsvolume.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <iostream>
#include <set>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <gpucast/math/parametric/nurbsvolume.hpp>

using namespace gpucast::math;




SUITE (nurbsvolume_class)
{


  float random (float min, float max)
  {
    float rel_value = float(rand())/RAND_MAX;
    return min + rel_value * fabs(max - min);
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
      knots_u.insert(random(0.1f, 0.9f));
      knots_u.insert(random(0.1f, 0.9f));
    }

    for (unsigned int v = 0; v != order_v; ++v)
    {
      knots_v.insert(0.0f);
      knots_v.insert(1.0f);
      knots_v.insert(random(0.1f, 0.9f));
      knots_v.insert(random(0.1f, 0.9f));
      knots_v.insert(random(0.1f, 0.9f));
    }

    for (unsigned int w = 0; w != order_w; ++w)
    {
      knots_w.insert(0.0f);
      knots_w.insert(1.0f);
      knots_w.insert(random(0.1f, 0.9f));
    }

    nv.knotvector_u(knots_u.begin(), knots_u.end());
    nv.knotvector_v(knots_v.begin(), knots_v.end());
    nv.knotvector_w(knots_w.begin(), knots_w.end());


    nv.resize(knots_u.size() - order_u,
              knots_v.size() - order_v,
              knots_w.size() - order_w);

    for (unsigned int w = 0; w != knots_w.size() - order_w; ++w) {
      for (unsigned int v = 0; v != knots_v.size() - order_v; ++v) {
        for (unsigned int u = 0; u != knots_u.size() - order_u; ++u) {
          point3d p(u + random(-0.5f, 0.5f), v + random(-0.5f, 0.5f), w + random(-0.5f, 0.5f));
          nv.set_point(u,v,w,p);
        }
      }
    }

    return nv;
  }

  TEST(default_ctor)
  {
    create_volume();
  }

}

