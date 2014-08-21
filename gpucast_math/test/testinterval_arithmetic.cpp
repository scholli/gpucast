/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testbeziersurface.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <gpucast/math/parametric/beziersurface.hpp>

using namespace gpucast::math;

typedef point<interval<double>, 3> ipoint3d;

SUITE(interval_arithmetic)
{

  beziersurface<ipoint3d> create_interval_beziersurface()
  {
    beziersurface<ipoint3d> bs;

    bs.degree_u(2);
    bs.degree_v(3);

    bs.add(ipoint3d(0.0, 0.0, 0.0, 1.0));   // first row
    bs.add(ipoint3d(0.1, 1.0, 2.0, 3.0));   // first row
    bs.add(ipoint3d(0.2, 2.2, 1.0, 1.0));   // first row
    bs.add(ipoint3d(0.0, 3.0, 1.0, 1.0));   // first row

    bs.add(ipoint3d(1.0, 0.0, 3.0, 2.0));   // second row
    bs.add(ipoint3d(1.0, 1.1, 1.0, 3.0));   // second row
    bs.add(ipoint3d(1.3, 2.5, 2.0, 1.0));   // second row
    bs.add(ipoint3d(1.0, 3.4, 4.0, 1.0));   // second row

    bs.add(ipoint3d(2.0, 0.0, 2.0, 1.0));   // third row
    bs.add(ipoint3d(2.3, 1.0, 4.0, 2.0));   // third row
    bs.add(ipoint3d(2.1, 2.3, 0.0, 1.0));   // third row
    bs.add(ipoint3d(2.0, 3.1, 1.0, 1.0));   // third row

    return bs;
  }

  beziersurface<point3d> create_beziersurface()
  {
    beziersurface<point3d> bs;

    bs.degree_u(2);
    bs.degree_v(3);

    bs.add(point3d(0.0, 0.0, 0.0, 1.0));   // first row
    bs.add(point3d(0.1, 1.0, 2.0, 3.0));   // first row
    bs.add(point3d(0.2, 2.2, 1.0, 1.0));   // first row
    bs.add(point3d(0.0, 3.0, 1.0, 1.0));   // first row

    bs.add(point3d(1.0, 0.0, 3.0, 2.0));   // second row
    bs.add(point3d(1.0, 1.1, 1.0, 3.0));   // second row
    bs.add(point3d(1.3, 2.5, 2.0, 1.0));   // second row
    bs.add(point3d(1.0, 3.4, 4.0, 1.0));   // second row

    bs.add(point3d(2.0, 0.0, 2.0, 1.0));   // third row
    bs.add(point3d(2.3, 1.0, 4.0, 2.0));   // third row
    bs.add(point3d(2.1, 2.3, 0.0, 1.0));   // third row
    bs.add(point3d(2.0, 3.1, 1.0, 1.0));   // third row

    return bs;
  }

  TEST(interval_plus)
  {
    intervald a(0.3, 0.5);
    intervald b(0.1, 0.8);

    auto c = a + b;

    CHECK_CLOSE(c.minimum(), 0.4, 0.0001);
    CHECK_CLOSE(c.maximum(), 1.3, 0.0001);
  }
  
  TEST(interval_minus)
  {
    intervald a(0.3, 0.5);
    intervald b(0.1, 0.8);

    auto c = a - b;

    CHECK_CLOSE(c.minimum(), -0.5, 0.0001);
    CHECK_CLOSE(c.maximum(), 0.4, 0.0001);
  }

  TEST(interval_times)
  {
    intervald a(0.3, 0.5);
    intervald b(0.1, 0.8);

    auto c = a * b;

    CHECK_CLOSE(c.minimum(), 0.03, 0.0001);
    CHECK_CLOSE(c.maximum(), 0.4, 0.0001);
  }

  TEST(interval_divi)
  {
    intervald a(0.3, 0.5);
    intervald b(0.1, 0.8);

    auto c = a * intervald(1.0 / b.maximum(), 1.0 / b.minimum());
    auto d = a / b;

    CHECK_CLOSE(c.minimum(), d.minimum(), 0.0001);
    CHECK_CLOSE(c.maximum(), d.maximum(), 0.0001);
  }


  TEST(testing_interval)
  {
#if 0
    horner<ipoint3d> h3d;
    decasteljau<ipoint3d> d3d;

    horner<point3d> horner;
    decasteljau<point3d> decasteljau;

    auto b = create_beziersurface();
    auto bs = create_interval_beziersurface();
    
    auto u = interval<double>(0.3, 0.4);
    auto v = interval<double>(0.4, 0.6);

    auto p = bs.evaluate(u, v, h3d);
    auto p2 = bs.evaluate(u, v, d3d);
    
    std::cout << "u = [0.3,0.4], v = [0.4,0.6], interval horner evaluation : " << p << std::endl;
    std::cout << "u = [0.3,0.4], v = [0.4,0.6], interval decasteljau evaluation : " << p2 << std::endl;

    auto u2 = interval<double>(0.3, 0.3000001);
    auto v2 = interval<double>(0.4, 0.4000001);

    auto p3 = bs.evaluate(u2, v2, d3d);
    auto p4 = b.evaluate(0.3, 0.4, decasteljau);

    std::cout << "u = [0.3,0.3+e], v = [0.4,0.4+e], interval decasteljau evaluation : " << p3 << std::endl;
    std::cout << "u = 0.3, v = 0.4, normal decasteljau evaluation : " << p4 << std::endl;

    auto p5 = bs.evaluate(u2, v2, h3d);
    auto p6 = b.evaluate(0.3, 0.4, horner);

    std::cout << "u = [0.3,0.3+e], v = [0.4,0.4+e], interval horner evaluation : " << p5 << std::endl;
    std::cout << "u = 0.3, v = 0.4, normal horner evaluation : " << p6 << std::endl;

    int splits = 100;

    int cells_converged = 0;

    for (int iu = 1; iu < splits; ++iu) {
      for (int iv = 1; iv < splits; ++iv) {
        double umin = double(iu) / splits;
        double umax = double(iu + 1) / splits;
        double vmin = double(iv) / splits;
        double vmax = double(iv + 1) / splits;

        umin = std::max(umin, 1.0/(splits+1));
        vmin = std::max(vmin, 1.0 / (splits + 1));

        auto urange = interval<double>(umin, umax);
        auto vrange = interval<double>(vmin, vmax);

        point<interval<double>, 3> puv, du, dv;
        try {
          bs.evaluate(urange, vrange, puv, du, dv, h3d);
        }
        catch ( ...) {
          std::cout << "error evaluating " << urange << " : " << vrange << std::endl;
        }

        point3d du_bboxmin(du[0].minimum(),
                        du[1].minimum(),
                        du[2].minimum());

        point3d du_bboxmax(du[0].maximum(),
                           du[1].maximum(),
                           du[2].maximum());

        point3d dv_bboxmin(dv[0].minimum(),
                           dv[1].minimum(),
                           dv[2].minimum());

        point3d dv_bboxmax(dv[0].maximum(),
                           dv[1].maximum(),
                           dv[2].maximum());

        axis_aligned_boundingbox<point3d> bbox_du(du_bboxmin, du_bboxmax);
        axis_aligned_boundingbox<point3d> bbox_dv(dv_bboxmin, dv_bboxmax);

        point3d origin(0.0, 0.0, 0.0);

        if (!bbox_du.is_inside(origin) &&
            !bbox_dv.is_inside(origin)) {
          cells_converged++;
          //std::cout << "interval converges. " << std::endl;
        }
        else {
          //std::cout << "interval does not converge. " << std::endl;
        }

      }
    }

    std::cout << "cells converged : " << 100*cells_converged / (splits*splits) << "%\n";

    CHECK(bs.size() == 12);
#endif
  }




}
