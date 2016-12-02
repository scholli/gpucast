/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testpoint.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <vector>
#include <gpucast/math/matrix.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/math/parametric/point.hpp>

using namespace gpucast::math;

SUITE (matrix)
{

  matrix<double, 4,4> generate_matrix4x4(double min, double max)
  {
    double d = max - min;

    matrix<double, 4, 4> m;
    for (std::size_t r = 0; r != 4; ++r)
    {
      for (std::size_t c = 0; c != 4; ++c)
      {
        m[r][c] = min + d * double(rand()) / RAND_MAX;

        // try to avoid numerical problems
        if (fabs(m[r][c]) < 0.01) {
          m[r][c] = 0.01;
        }
      }
    }

    return m;
  }

  matrix<double, 3, 3> generate_matrix3x3 ( double min, double max )
  {
    double d = max - min;

    matrix<double, 3, 3> m;
    for (std::size_t r = 0; r != 3; ++r)
    {
      for (std::size_t c = 0; c != 3; ++c)
      {
        m[r][c] = min + d * double(rand())/RAND_MAX;

        // try to avoid numerical problems
        if ( fabs(m[r][c]) < 0.01) {
          m[r][c] = 0.01;
        }
      }
    }

    return m;
  }



  matrix<double, 2, 2> generate_matrix2x2 ( double min, double max )
  {
    double d = max - min;

    matrix<double, 2, 2> m;
    for (std::size_t r = 0; r != 2; ++r)
    {
      for (std::size_t c = 0; c != 2; ++c)
      {
        m[r][c] = min + d * double(rand())/RAND_MAX;

        // try to avoid numerical problems
        if ( fabs(m[r][c]) < 0.01) {
          m[r][c] = 0.01;
        }
      }
    }

    return m;
  }


  TEST(default_ctor)
  {
    matrix<float, 9, 12> m;
  }

  TEST(identity_matrix)
  {
    matrix<float, 2, 2> m2;
    CHECK(m2[0][0] == 1.0f);
    CHECK(m2[0][1] == 0.0f);
    CHECK(m2[1][0] == 0.0f);
    CHECK(m2[1][1] == 1.0f);

    matrix<float, 3, 3> m3;
    CHECK(m3[0][0] == 1.0f);
    CHECK(m3[0][1] == 0.0f);
    CHECK(m3[0][2] == 0.0f);
    CHECK(m3[1][0] == 0.0f);
    CHECK(m3[1][1] == 1.0f);
    CHECK(m3[1][2] == 0.0f);
    CHECK(m3[2][0] == 0.0f);
    CHECK(m3[2][1] == 0.0f);
    CHECK(m3[2][2] == 1.0f);

    matrix<float, 4, 4> m4;
    CHECK(m4[0][0] == 1.0f);
    CHECK(m4[1][1] == 1.0f);
    CHECK(m4[2][2] == 1.0f);
    CHECK(m4[3][3] == 1.0f);
    CHECK(m4[3][2] == 0.0f);
    CHECK(m4[1][2] == 0.0f);
    CHECK(m4[0][2] == 0.0f);
    // ..
  }

  TEST(operator_dot)
  {
    matrix<float, 4, 4> m;
    point<float, 4> p;
    point<float, 4> p2 = m * p;
  }

  TEST(access_operator)
  {
    matrix<float, 9, 12> m;

    m[5][2]   = 5.0f;
    CHECK(m[5][2] == 5.0f);
  }

  TEST(submatrix)
  {
    matrix<float, 4, 4> m4;
    matrix<float, 3, 3> I3;
    matrix<float, 3, 3> m0 = generate_submatrix(m4, 0, 0);
    matrix<float, 3, 3> m1 = generate_submatrix(m4, 1, 1);
    matrix<float, 3, 3> m2 = generate_submatrix(m4, 2, 2);
    matrix<float, 3, 3> m3 = generate_submatrix(m4, 2, 1);
    matrix<float, 3, 3> m5 = generate_submatrix(m4, 0, 1);
    matrix<float, 3, 3> m6 = generate_submatrix(m4, 2, 0);

    CHECK (I3 == m0);
    CHECK (I3 == m1);
    CHECK (I3 == m2);
    CHECK (I3 != m3);
    CHECK (I3 != m5);
    CHECK (I3 != m6);
  }

  TEST(eigenvalues)
  {
    matrix<double, 3, 3> A;
    A[0][0] = 1;
    A[0][1] = 0;
    A[0][2] = 1;

    A[1][0] = 2;
    A[1][1] = 2;
    A[1][2] = 1;

    A[2][0] = 4;
    A[2][1] = 2;
    A[2][2] = 1;

    std::set<double> roots = A.eigenvalues(-100, 100, 0.00001);

    std::set<double>::const_iterator i = roots.begin();
    CHECK_CLOSE(*i++, -1, 0.000001);
    CHECK_CLOSE(*i++,  1, 0.000001);
    CHECK_CLOSE(*i,  4, 0.000001);
  }


  TEST(eigenvectors)
  {
    matrix<double, 3, 3> A;
    A[0][0] = 1;
    A[0][1] = 0;
    A[0][2] = 1;

    A[1][0] = 2;
    A[1][1] = 2;
    A[1][2] = 1;

    A[2][0] = 4;
    A[2][1] = 2;
    A[2][2] = 1;

    matrix<double, 3, 3> iA = A.inverse();
    matrix<double, 3, 3> I = A * iA;

    double const eps = 0.0000001;
    double const minval = -1000.0;
    double const maxval =  1000.0;

    std::set<double> ev_set = A.eigenvalues ( minval, maxval, eps );
    std::vector<double>             eigenvalues (ev_set.begin(), ev_set.end());
    std::vector<point<double, 3> >  eigenvectors = A.eigenvectors ( minval, maxval, eps );

    CHECK ( eigenvectors.size() == 3 );
    CHECK ( eigenvalues.size() == 3 );

    for ( unsigned i = 0; i != eigenvalues.size(); ++ i )
    {
      CHECK_CLOSE ( (A * eigenvectors[i])[0], (eigenvalues[i] * eigenvectors[i])[0], eps);
      CHECK_CLOSE ( (A * eigenvectors[i])[1], (eigenvalues[i] * eigenvectors[i])[1], eps);
      CHECK_CLOSE ( (A * eigenvectors[i])[2], (eigenvalues[i] * eigenvectors[i])[2], eps);
    }

    unsigned const MAX_TEST = 1000;
    for (unsigned i = 0; i != MAX_TEST; ++i )
    {
      matrix<double, 3, 3> A2 = generate_matrix3x3(-10.0, 10.0);

      std::set<double> ev_set2 = A2.eigenvalues ( minval, maxval, eps );
      std::vector<double>             eigenvalues2 (ev_set2.begin(), ev_set2.end());
      std::vector<point<double, 3> >  eigenvectors2 = A2.eigenvectors ( minval, maxval, eps );

      for ( unsigned i = 0; i != eigenvalues2.size(); ++i )
      {
        for ( unsigned k = 0; k != 3; ++k )
        {
          double lhs = (A2 * eigenvectors2[i])[k];
          double rhs = (eigenvalues2[i] * eigenvectors2[i])[k];
          CHECK_CLOSE ( lhs, rhs, 10*eps);
        }
      }
    }
  }


  TEST(inverse)
  {
    unsigned const MAX_TEST = 100;
    for (unsigned i = 0; i != MAX_TEST; ++i )
    {
      matrix<double, 3, 3> A    = generate_matrix3x3(-10.0, 10.0);
      matrix<double, 3, 3> invA = A.inverse();

      matrix<double, 3, 3> I0 = A * invA;
      matrix<double, 3, 3> I1 = invA * A;

      for (unsigned r = 0; r != 3; ++r )
      {
        for (unsigned c = 0; c != 3; ++c )
        {
          if ( r==c )
          {
            CHECK_CLOSE(I0[r][c], 1.0, 0.000001);
            CHECK_CLOSE(I1[r][c], 1.0, 0.000001);
          } else {
            CHECK_CLOSE(I0[r][c], 0.0, 0.000001);
            CHECK_CLOSE(I1[r][c], 0.0, 0.000001);
          }
        }
      }
    }


  matrix<double, 2, 2> M;
  M[0][0] = 2;
  M[0][1] = 4;
  M[1][0] = 1;
  M[1][1] = 5;

  matrix<double, 2, 2> invM = M.inverse();

  matrix<double, 2, 2> I = M * invM;

    for (unsigned i = 0; i != MAX_TEST; ++i )
    {
      matrix<double, 2, 2> A    = generate_matrix2x2(-10.0, 10.0);
      matrix<double, 2, 2> invA = A.inverse();

      matrix<double, 2, 2> I0 = A * invA;
      matrix<double, 2, 2> I1 = invA * A;

      for (unsigned r = 0; r != 2; ++r )
      {
        for (unsigned c = 0; c != 2; ++c )
        {
          if ( r==c )
          {
            CHECK_CLOSE(I0[r][c], 1.0, 0.000001);
            CHECK_CLOSE(I1[r][c], 1.0, 0.000001);
          } else {
            CHECK_CLOSE(I0[r][c], 0.0, 0.000001);
            CHECK_CLOSE(I1[r][c], 0.0, 0.000001);
          }
        }
      }
    }
  }

  TEST(matrix_determinant3x3)
  {
    std::vector<double> mat_data = { 1.0,  0.4, -0.3, 
                                     1.6,  0.3, -0.1, 
                                     -1.4,  0.4,  0.6 };
    matrix<double, 3, 3> m(mat_data.begin(), mat_data.end());
    auto d = m.determinant();
    CHECK_CLOSE(d, -0.426, 0.0001);
  }

  TEST(matrix_conversion)
  {
    std::vector<double> mat_data = { 1.0,  0.4, -0.3, 2.1,
                                     1.6,  0.3, -0.1, 0.1,
                                    -1.4,  0.4,  0.6, 1.1,
                                     0.2, -1.4, -0.7, 0.5 };

    matrix<double, 4, 4> m4x4 (mat_data.begin(), mat_data.end());
    matrix<double, 4, 4> m4x4t = m4x4.transpose();

    const double expected_result = -2.0508;

    point<double, 4> p0(1.0, 3.0, 2.0, 1.0);

    gpucast::math::matrix4d m4d(m4x4);
    gpucast::math::matrix4d m4dt(m4x4t);

    CHECK_CLOSE(m4x4.determinant(), expected_result, 0.0001);
    CHECK_CLOSE(m4x4t.determinant(), expected_result, 0.0001);

    CHECK_CLOSE(m4d.determinant(), expected_result, 0.0001);
    CHECK_CLOSE(m4dt.determinant(), expected_result, 0.0001);

    CHECK(m4x4 * p0 == m4d * p0);
    CHECK(m4x4t * p0 == m4dt * p0);
  }
}
