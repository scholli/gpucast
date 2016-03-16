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
#include <vector>

// header, project


namespace gpucast { namespace gl {

struct hullvertexmap
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

  inline hullvertexmap() {
    // table from Dieter Schmalstieg and Robert Tobler: Fast projected area computation for 3D bounding boxes [2005] 
    data.resize(44);

    data[0] = visible_vertex_set{ 0, 0, 0, 0, 0, 0, 0, 0 }; // inside
    data[1] = visible_vertex_set{ 1, 4, 0, 4, 7, 3, 0, 0 }; // left
    data[2] = visible_vertex_set{ 2, 4, 1, 2, 6, 5, 0, 0 }; // right
    data[3] = visible_vertex_set{ 3, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[4] = visible_vertex_set{ 4, 4, 0, 1, 5, 4, 0, 0 }; // bottom
    data[5] = visible_vertex_set{ 5, 6, 0, 1, 5, 4, 7, 3 }; // bottom, left
    data[6] = visible_vertex_set{ 6, 6, 0, 1, 2, 6, 5, 4 }; // bottom, right
    data[7] = visible_vertex_set{ 7, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[8] = visible_vertex_set{ 8, 4, 2, 3, 7, 6, 0, 0 }; // top
    data[9] = visible_vertex_set{ 9, 6, 4, 7, 6, 2, 3, 0 }; // top, left
    data[10] = visible_vertex_set{ 10, 6, 2, 3, 7, 6, 5, 1 }; // top, right
    data[11] = visible_vertex_set{ 11, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[12] = visible_vertex_set{ 12, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[13] = visible_vertex_set{ 13, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[14] = visible_vertex_set{ 14, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[15] = visible_vertex_set{ 15, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[16] = visible_vertex_set{ 16, 4, 0, 3, 2, 1, 0, 0 }; // front
    data[17] = visible_vertex_set{ 17, 6, 0, 4, 7, 3, 2, 1 }; // front, left
    data[18] = visible_vertex_set{ 18, 6, 0, 3, 2, 6, 5, 1 }; // front, right
    data[19] = visible_vertex_set{ 19, 0, 0, 0, 0, 0, 0, 0 }; // 
    data[20] = visible_vertex_set{ 20, 6, 0, 3, 2, 1, 5, 4 }; // front, bottom
    data[21] = visible_vertex_set{ 21, 6, 2, 1, 5, 4, 7, 3 }; // front, bottom, left
    data[22] = visible_vertex_set{ 22, 6, 0, 3, 2, 6, 5, 4 }; // front, bottom, right
    data[23] = visible_vertex_set{ 23, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[24] = visible_vertex_set{ 24, 6, 0, 3, 7, 6, 2, 1 }; // front, top
    data[25] = visible_vertex_set{ 25, 6, 0, 4, 7, 6, 2, 1 }; // front, top, left
    data[26] = visible_vertex_set{ 26, 6, 0, 3, 7, 6, 5, 1 }; // front, top, right
    data[27] = visible_vertex_set{ 27, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[28] = visible_vertex_set{ 28, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[29] = visible_vertex_set{ 29, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[30] = visible_vertex_set{ 30, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[31] = visible_vertex_set{ 31, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[32] = visible_vertex_set{ 32, 4, 4, 5, 6, 7, 0, 0 }; // back
    data[33] = visible_vertex_set{ 33, 6, 4, 5, 6, 7, 3, 0 }; // back, left
    data[34] = visible_vertex_set{ 34, 6, 1, 2, 6, 7, 4, 5 }; // back, right
    data[35] = visible_vertex_set{ 35, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[36] = visible_vertex_set{ 36, 6, 0, 1, 5, 6, 7, 4 }; // back, bottom
    data[37] = visible_vertex_set{ 37, 6, 0, 1, 5, 6, 7, 3 }; // back, bottom, left
    data[38] = visible_vertex_set{ 38, 6, 0, 1, 2, 6, 7, 4 }; // back, bottom, right
    data[39] = visible_vertex_set{ 39, 0, 0, 0, 0, 0, 0, 0 }; // -
    data[40] = visible_vertex_set{ 40, 6, 2, 3, 7, 4, 5, 6 }; // back, top, 
    data[41] = visible_vertex_set{ 41, 6, 0, 4, 5, 6, 2, 3 }; // back, top, left
    data[42] = visible_vertex_set{ 42, 6, 1, 2, 3, 7, 4, 5 }; // back, top, right
    data[43] = visible_vertex_set{ 43, 0, 0, 0, 0, 0, 0, 0 };
  }

  //d::vector<visible_vertex_set>::iterator begin()

  std::vector<visible_vertex_set> data;

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_UTIL_HULLVERTEXMAP_HPP
