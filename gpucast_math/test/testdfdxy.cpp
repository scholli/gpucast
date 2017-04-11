/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testdfdxy.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <cstdlib>

#include <gpucast/math/vec4.hpp>
#include <gpucast/math/vec2.hpp>

typedef gpucast::math::vec4d vec4;
typedef gpucast::math::vec2d vec2;

void compute_partial_derivatives(vec4 a, // xy_uv
  vec4 b, // xy_uv
  vec4 c, // xy_uv
  vec2& duv_dx,
  vec2& duv_dy)
{
  vec2 ab(b[0] - a[0], b[1] - a[1]);
  vec2 ac(c[0] - a[0], c[1] - a[1]);

  float du_dy = (b[2] / ab[0] - c[2] / ac[0] - a[2] / ab[0] + a[2] / ac[0]) /
    (ab[1] / ab[0] - ac[1] / ac[0]);
  float dv_dy = (b[3] / ab[0] - c[3] / ac[0] - a[3] / ab[0] + a[3] / ac[0]) /
    (ab[1] / ab[0] - ac[1] / ac[0]);

  float du_dx = (b[2] - a[2] - ab[1] * du_dy) / ab[0];
  float dv_dx = (b[3] - a[3] - ab[1] * dv_dy) / ab[0];

  duv_dx = vec2(du_dx, dv_dx);
  duv_dy = vec2(du_dy, dv_dy);
}



SUITE (dFdxy)
{

  TEST(clip)
  {
    vec4 A = vec4{ 0.6, 1.3, 0, 1 };
    vec4 B = vec4{ 4.8, 5.7, 1, 1 };
    vec4 C = vec4{ 6.3, 3, 1, 0 };
    vec2 duvdx, duvdy;

    double tolerance = 0.00001;

    compute_partial_derivatives(B, C, A, duvdx, duvdy);
    CHECK_CLOSE(B[2], A[2] + (B[0] - A[0]) * duvdx[0] + (B[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(B[3], A[3] + (B[0] - A[0]) * duvdx[1] + (B[1] - A[1]) * duvdy[1], tolerance);
    CHECK_CLOSE(C[2], A[2] + (C[0] - A[0]) * duvdx[0] + (C[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(C[3], A[3] + (C[0] - A[0]) * duvdx[1] + (C[1] - A[1]) * duvdy[1], tolerance);

    compute_partial_derivatives(A, B, C, duvdx, duvdy);
    CHECK_CLOSE(B[2], A[2] + (B[0] - A[0]) * duvdx[0] + (B[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(B[3], A[3] + (B[0] - A[0]) * duvdx[1] + (B[1] - A[1]) * duvdy[1], tolerance);
    CHECK_CLOSE(C[2], A[2] + (C[0] - A[0]) * duvdx[0] + (C[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(C[3], A[3] + (C[0] - A[0]) * duvdx[1] + (C[1] - A[1]) * duvdy[1], tolerance);

    compute_partial_derivatives(C, B, A, duvdx, duvdy);
    CHECK_CLOSE(B[2], A[2] + (B[0] - A[0]) * duvdx[0] + (B[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(B[3], A[3] + (B[0] - A[0]) * duvdx[1] + (B[1] - A[1]) * duvdy[1], tolerance);
    CHECK_CLOSE(C[2], A[2] + (C[0] - A[0]) * duvdx[0] + (C[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(C[3], A[3] + (C[0] - A[0]) * duvdx[1] + (C[1] - A[1]) * duvdy[1], tolerance);

    compute_partial_derivatives(B, A, C, duvdx, duvdy);
    CHECK_CLOSE(B[2], A[2] + (B[0] - A[0]) * duvdx[0] + (B[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(B[3], A[3] + (B[0] - A[0]) * duvdx[1] + (B[1] - A[1]) * duvdy[1], tolerance);
    CHECK_CLOSE(C[2], A[2] + (C[0] - A[0]) * duvdx[0] + (C[1] - A[1]) * duvdy[0], tolerance);
    CHECK_CLOSE(C[3], A[3] + (C[0] - A[0]) * duvdx[1] + (C[1] - A[1]) * duvdy[1], tolerance);
  }
}
