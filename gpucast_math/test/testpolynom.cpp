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
#include <iostream>

#include <gpucast/math/polynomial.hpp>
#include <gpucast/math/matrix.hpp>
#include <gpucast/math/binomial_coefficient.hpp>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>

using namespace gpucast::math;

namespace
{

  polynomial<double> generate4d ( double min, double max )
  {
    double d = max - min;

    polynomial<double> p;
    p.order(4);

    for (std::size_t i = 0; i != 4; ++i) 
    {
      p[i] = min + d * double(rand())/RAND_MAX;

      // try to avoid numerical problems 
      if ( fabs(p[i]) < 0.01) {
        p[i] = 0.01;
      }
    }

    return p;
  }

}

SUITE (polynomial)
{

  TEST(ctor)
  {
    polynomial<double> p = generate4d(-10.0, 10.0);
    std::cout << p << std::endl;
  }

  TEST(print)
  {
    polynomial<double> p = generate4d(-10.0, 10.0);
    std::cout << p << std::endl;
  }

  TEST(derive)
  {
    polynomial<double> p    = generate4d(-10.0, 10.0);
    polynomial<double> dp   = derive(p); 
    polynomial<double> ddp  = derive(dp);
    polynomial<double> dddp = derive(ddp);
  }

  TEST(solve)
  {
    std::size_t max_runs    = 1000;
    double const tolerance  = 0.000000001;

    for (std::size_t i = 0; i != max_runs; ++i) 
    {
      polynomial<double> p    = generate4d(-1.0, 1.0);
      std::set<double> roots = p.solve(-100.0, 100.0, tolerance);

      CHECK ( roots.size() == 1 || roots.size() == 3 );

      BOOST_FOREACH(double root, roots) 
      {        
        CHECK_CLOSE( p.evaluate(root), 0, tolerance);
      }
    }
  }


  TEST(operatork)
  {
    polynomial<double> p(2, -3);
    p *= 3;

    polynomial<double> p2(6, -9);

    CHECK(p == p2);
    CHECK(p[0] == 6);
    CHECK(p[1] == -9);

    polynomial<double> p3 = p*p;
    CHECK(p3[0] == 36);
    CHECK(p3[1] == -108);
    CHECK(p3[2] == 81);

    polynomial<double> p4(1, -2);
    p4 *= p4;
    CHECK(p4[0] == 1);
    CHECK(p4[1] == -4);
    CHECK(p4[2] == 4);

  }


  TEST(matrx)
  {
    polynomial<double> p;
    matrix<polynomial<double>, 3, 3> I;

    polynomial<double> lambda(0, 1);

    matrix<polynomial<double>, 3, 3> A;
    A[0][0] = 1;
    A[0][1] = 0;
    A[0][2] = 1;

    A[1][0] = 2;
    A[1][1] = 2;
    A[1][2] = 1;

    A[2][0] = 4;
    A[2][1] = 2;
    A[2][2] = 1;

    matrix<polynomial<double>, 3, 3> C = lambda * I - A;
    polynomial<double> D = C.determinant();
    std::set<double> roots = D.solve(-100.0, 100.0, 0.000001);

    std::set<double>::const_iterator i = roots.begin();
    CHECK(roots.size() == 3);
    CHECK_CLOSE(*i++, -1, 0.000001);
    CHECK_CLOSE(*i++,  1, 0.000001);
    CHECK_CLOSE(*i,  4, 0.000001);
  }

  TEST ( pow )
  {
    polynomial<double> p(2, 3);
    polynomial<double> p4 = gpucast::math::pow ( p, 4 );
    CHECK ( p4[0] == 16 );
    CHECK ( p4[1] == 96 );
    CHECK ( p4[2] == 216 );
    CHECK ( p4[3] == 216 );
    CHECK ( p4[4] == 81 );
  }



}



