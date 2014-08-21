/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : test/testpointmesh3d.cpp
*  project    : tml
*  description:
*
********************************************************************************/

#include <fstream>

#include <unittest++/UnitTest++.h>

#include <gpucast/math/parametric/pointmesh3d.hpp>

using namespace gpucast::math;

SUITE (pointmesh3d_class)
{
  TEST(ctor_dtor_copy)
  {
    std::size_t const width  = 3;
    std::size_t const height = 4;
    std::size_t const depth  = 5;

    std::vector<point3d> data(width * height * depth, point3d(1.0, 1.0, 2.0));

    pointmesh3d<point3d> mesh0(width, height, depth);
    pointmesh3d<point3d> mesh1(data.begin(), data.end(), width, height, depth);

    CHECK(mesh0.width ()  == width);
    CHECK(mesh0.height() == height);
    CHECK(mesh0.depth ()  == depth);

    CHECK(mesh1.width ()  == width);
    CHECK(mesh1.height() == height);
    CHECK(mesh1.depth ()  == depth);

    pointmesh3d<point3d>::const_iterator b = mesh0.begin();
    while (b != mesh0.end())
    {
      CHECK_EQUAL(*b, point3d(0.0, 0.0, 0.0));
      ++b;
    }

    b = mesh1.begin();
    while (b != mesh1.end())
    {
      CHECK_EQUAL(*b, point3d(1.0, 1.0, 2.0));
      ++b;
    }
  }

  TEST(check_slice_function)
  {
    std::size_t width  = 4;
    std::size_t height = 6;
    std::size_t depth  = 5;

    pointmesh3d<point3d> mesh;
    for (std::size_t z = 0; z != depth; ++z)
    {
      for (std::size_t y = 0; y != height; ++y)
      {
        for (std::size_t x = 0; x != width; ++x)
        {
          point3d p = point3d(double(x), double(y), double(z));
          mesh.push_back(p);
        }
      }
    }

    mesh.width (width);
    mesh.height(height);
    mesh.depth (depth);

    pointmesh2d<point3d> m0 = mesh.submesh(1, 5);
    pointmesh2d<point3d> m1 = mesh.submesh(0, 1);
    pointmesh2d<point3d> m2 = mesh.submesh(2, 3);

    for (pointmesh2d<point3d>::const_iterator i = m0.begin(); i != m0.end(); ++i)
    {
      CHECK_EQUAL((*i)[point3d::y], 5);
    }
    for (pointmesh2d<point3d>::const_iterator i = m1.begin(); i != m1.end(); ++i)
    {
      CHECK_EQUAL((*i)[point3d::x], 1);
    }
    for (pointmesh2d<point3d>::const_iterator i = m2.begin(); i != m2.end(); ++i)
    {
      CHECK_EQUAL((*i)[point3d::z], 3);
    }
  }

  TEST(transpose)
  {
    std::size_t width  = 4;
    std::size_t height = 6;
    std::size_t depth  = 5;

    pointmesh3d<point3d> mesh;
    mesh.width (width);
    mesh.height(height);
    mesh.depth (depth);

    for (std::size_t z = 0; z != mesh.depth(); ++z)
    {
      for (std::size_t y = 0; y != mesh.height(); ++y)
      {
        for (std::size_t x = 0; x != mesh.width(); ++x)
        {
          point3d p = point3d(double(x), double(y), double(z));
          mesh.push_back(p);
        }
      }
    }

    // 0 -> 2
    // 1 -> 0
    // 2 -> 1

    mesh.transpose(2, 0, 1); // 6,5,4

    for (std::size_t z = 0; z != mesh.depth(); ++z)
    {
      for (std::size_t y = 0; y != mesh.height(); ++y)
      {
        for (std::size_t x = 0; x != mesh.width(); ++x)
        {
          CHECK_EQUAL(mesh(x,y,z), point3d(double(z),double(x),double(y)));
        }
      }
    }

    CHECK_EQUAL(mesh.width() , 6);
    CHECK_EQUAL(mesh.height(), 5);
    CHECK_EQUAL(mesh.depth() , 4);

    // 0 -> 1
    // 1 -> 2
    // 2 -> 0

    mesh.transpose(1, 2, 0); // 4,6,5

    for (std::size_t z = 0; z != mesh.depth(); ++z)
    {
      for (std::size_t y = 0; y != mesh.height(); ++y)
      {
        for (std::size_t x = 0; x != mesh.width(); ++x)
        {
          CHECK_EQUAL(mesh(x,y,z), point3d(double(x),double(y),double(z)));
        }
      }
    }
    CHECK_EQUAL(mesh.width() , 4);
    CHECK_EQUAL(mesh.height(), 6);
    CHECK_EQUAL(mesh.depth() , 5);
  }



  TEST(submesh)
  {
    std::size_t width  = 4;
    std::size_t height = 6;
    std::size_t depth  = 5;

    pointmesh3d<point3d> mesh;
    mesh.width (width);
    mesh.height(height);
    mesh.depth (depth);

    for (std::size_t z = 0; z != mesh.depth(); ++z)
    {
      for (std::size_t y = 0; y != mesh.height(); ++y)
      {
        for (std::size_t x = 0; x != mesh.width(); ++x)
        {
          point3d p = point3d(double(x), double(y), double(z));
          mesh.push_back(p);
        }
      }
    }

    std::vector<point3d> sm0 = mesh.submesh(0, 3, 2);
    std::vector<point3d> sm1 = mesh.submesh(1, 2, 1);
    std::vector<point3d> sm2 = mesh.submesh(2, 2, 4);

    for (unsigned int i = 0; i != width; ++i)
    {
      CHECK_EQUAL(sm0[i], point3d(double(i), 3.0, 2.0));
    }

    for (unsigned int i = 0; i != height; ++i)
    {
      CHECK_EQUAL(sm1[i], point3d(2.0, double(i), 1.0));
    }

    for (unsigned int i = 0; i != depth; ++i)
    {
      CHECK_EQUAL(sm2[i], point3d(2.0, 4.0, double(i)));
    }
  }

}
