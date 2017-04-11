/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/coverage.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <unittest++/UnitTest++.h>

#include <cstdlib>

#include <gpucast/math/util/prefilter2d.hpp>
#include <gpucast/math/vec4.hpp>
#include <gpucast/math/vec2.hpp>
#include <gpucast/math/matrix2x2.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziercurve.hpp>

using namespace gpucast::math;




vec2d tangent_to_gradient(vec2d tangent) {
  return vec2d(tangent[1], -tangent[0]);
}

vec2d get_closest(vec2d const& uv, vec2d const& bbox_min, vec2d const& bbox_max)
{
  return (uv - bbox_min).length_square() > (uv - bbox_max).length_square() ? bbox_max : bbox_min;
}

double gradient_to_tex_coord(vec2d gradient) {
  gradient.normalize();
  double sin_a = gradient[1] > 0.0 ? gradient[1] : -gradient[1];
  double gradient_angle_rad = asin(sin_a);
  double gradient_tex_coord = gradient_angle_rad / (2.0 * M_PI);
  return gradient_tex_coord;
}


double test_coverage(beziercurve2d const& bc,
  vec2d const& uv,
  vec2d const& duvdx,
  vec2d const& duvdy)
{
  horner<point2d> h;
  point2d px;
  auto bbox = bc.bbox_simple();
  bc.evaluate(0.5, px, h);
  vec2d closest_point_to_curve{ px[0], px[1] };

  matrix2d J = { duvdx[0], duvdx[1], duvdy[0], duvdy[1] };
  J.transpose();
  auto Jinv = J;
  Jinv.invert();

  // compute points of remaining bbox
  vec2d p0, pn;
  if (bc.is_increasing(point2d::u)) {
    p0 = bbox.min;
    pn = bbox.max;
  }
  else {
    p0 = vec2d{ bbox.max[0], bbox.min[1] };
    pn = vec2d{ bbox.min[0], bbox.max[1] };
  }

  vec2d p0_pixel = Jinv*p0; // ok
  vec2d pn_pixel = Jinv*pn; // ok
  vec2d uv_pixel_coords = Jinv*uv;; // ok
  vec2d point_pixel_coords = Jinv*closest_point_to_curve; // ok
  vec2d uv_to_point_pixel = point_pixel_coords - uv_pixel_coords;

  vec2d tangent_pixel = get_closest(uv_pixel_coords, p0_pixel, pn_pixel);
  vec2d normalized_closest_gradient = vec2d(tangent_pixel[1], -tangent_pixel[0]) / tangent_pixel.abs();
  const double gradient_angle_sin = normalized_closest_gradient[1] > 0.0 ? normalized_closest_gradient[1] : -normalized_closest_gradient[1];
  const double gradient_angle_rad = asin(gradient_angle_sin);
  const double gradient_tex_coord = gradient_angle_rad / (2.0 * M_PI);
  const double angle_degree = (360 * gradient_angle_rad) / (2.0 * M_PI);
  float distance_pixel_coords = abs(dot(normalized_closest_gradient, uv_to_point_pixel));

  std::cout << "px : " << px << std::endl;
  std::cout << "p0 : " << p0 << std::endl;
  std::cout << "pn : " << pn << std::endl;
  std::cout << "p0_pixel : " << p0_pixel << std::endl;
  std::cout << "pn_pixel : " << pn_pixel << std::endl;
  std::cout << "uv_pixel_coords : " << uv_pixel_coords << std::endl;
  std::cout << "point_pixel_coords : " << point_pixel_coords << std::endl;
  std::cout << "uv_to_point_pixel : " << uv_to_point_pixel << std::endl;

  std::cout << "Jinv : " << Jinv << std::endl;
  std::cout << "tangent_pixel : " << tangent_pixel << std::endl;
  std::cout << "normalized_closest_gradient : " << normalized_closest_gradient << std::endl;
  std::cout << "gradient_angle_cos : " << gradient_angle_sin << std::endl;
  std::cout << "gradient_angle_rad : " << gradient_angle_rad << std::endl;
  std::cout << "gradient_tex_coord : " << gradient_tex_coord << std::endl;
  std::cout << "angle_degree : " << angle_degree << std::endl;
  std::cout << "distance_pixel_coords : " << distance_pixel_coords << std::endl;

  util::prefilter2d<vec2d> prefilter(256);
  const double coverage_center_in = prefilter(vec2d(gradient_angle_rad, distance_pixel_coords));
  const double coverage_center_out = prefilter(vec2d(gradient_angle_rad, -distance_pixel_coords));
  const double tmp = prefilter(vec2d(0, -0.6));
  const double tmp2 = prefilter(vec2d(M_PI/4.0, -0.6));

  std::cout << "coverage_center_in : " << coverage_center_in << std::endl;
  std::cout << "coverage_center_out : " << coverage_center_out << std::endl;
  std::cout << "tmp : " << tmp << std::endl;
  std::cout << "tmp2 : " << tmp2 << std::endl;

  

  return 0.0;
}



SUITE (coverage_estimation)
{
  TEST(gradient_to_tex) {
    CHECK_CLOSE(gradient_to_tex_coord(vec2d(std::sqrt(2) / 2, std::sqrt(2) / 2)), 0.125, 0.00001);
    CHECK_CLOSE(gradient_to_tex_coord(vec2d(-std::sqrt(2) / 2, std::sqrt(2) / 2)), 0.125, 0.00001);
    CHECK_CLOSE(gradient_to_tex_coord(vec2d(0, 1)), 0.25, 0.00001);
    CHECK_CLOSE(gradient_to_tex_coord(vec2d(1, 0)), 0.0, 0.00001);
    CHECK_CLOSE(gradient_to_tex_coord(vec2d(-1, 0)), 0.0, 0.00001);
  }

  TEST(coverage_horizontally_increasing)
  {
    // create curve
    beziercurve2d bc1;

    point2d p0(1.0, 1.0, 1.0);
    point2d p1(1.5, 3.0, 1.0);
    point2d p2(3.0, 3.5, 1.0);

    bc1.add(p0);
    bc1.add(p1);
    bc1.add(p2);

    vec2d uv { 2.7, 2.5 };
    vec2d duvdx { 0.2, 1.0 };
    vec2d duvdy { -1.0, 1.5 };

    auto coverage1 = test_coverage(bc1, uv, duvdx, duvdy);

    beziercurve2d bc2;

    point2d p20(3.0, 1.0, 1.0);
    point2d p21(3.0, 3.0, 1.0);
    point2d p22(2.0, 4.0, 1.0);

    bc2.add(p20);
    bc2.add(p21);
    bc2.add(p22);

    auto coverage2 = test_coverage(bc2, uv, duvdx, duvdy);
  }
}
