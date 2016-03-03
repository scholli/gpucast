/********************************************************************************
*
* Copyright (C) 2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : hullvertexmap.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_UTIL_HULLVERTEXMAP_HPP
#define GPUCAST_GL_UTIL_HULLVERTEXMAP_HPP

// header, system
#include <array>
#include <vector>

// header, project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/eventhandler.hpp>
#include <gpucast/math/matrix4x4.hpp>



namespace gpucast { namespace gl {

struct GPUCAST_GL hullvertexmap
{
  // vertex indices:
  //       7______6
  //      /.     /|
  //     / .    / |
  //    3------2  |           +y   +z
  //    |  4. .| ./5           |  /
  //    | .    | /             | /
  //    --------/              ----- +x
  //    0      1

  struct visible_vertex_set 
  {
    unsigned char index;
    unsigned char num_visible_vertices;
    unsigned char visible_vertex0;
    unsigned char visible_vertex1;
    unsigned char visible_vertex2;
    unsigned char visible_vertex3;
    unsigned char visible_vertex4;
    unsigned char visible_vertex5;
  };

  hullvertexmap();

  std::vector<visible_vertex_set> data;

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_UTIL_HULLVERTEXMAP_HPP
