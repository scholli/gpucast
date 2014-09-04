/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : cube.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_CUBE_HPP
#define GPUCAST_GL_CUBE_HPP

#include <vector>

#include <gpucast/gl/glpp.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/math/vec4.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL cube
{
public : // c'tor/d'tor

  cube             ( GLint vertexattrib_index   = 0,
                     GLint colorattrib_index    = 1,
                     GLint normalattrib_index   = 2,
                     GLint texcoordattrib_index = 3 );

  ~cube            ( );

  cube(cube const&  other) = delete;
  cube operator=(cube const&  other) = delete;

public : // methods

  void draw            ( bool wireframe = false );

  void set_vertices    ( gpucast::math::vec4f const& lbf = gpucast::math::vec4f(-1.0f, -1.0f, -1.0f, 1.0f),
                         gpucast::math::vec4f const& rbf = gpucast::math::vec4f( 1.0f, -1.0f, -1.0f, 1.0f),
                         gpucast::math::vec4f const& ltf = gpucast::math::vec4f(-1.0f,  1.0f, -1.0f, 1.0f),
                         gpucast::math::vec4f const& rtf = gpucast::math::vec4f( 1.0f,  1.0f, -1.0f, 1.0f),
                         gpucast::math::vec4f const& lbr = gpucast::math::vec4f(-1.0f, -1.0f,  1.0f, 1.0f),
                         gpucast::math::vec4f const& rbr = gpucast::math::vec4f( 1.0f, -1.0f,  1.0f, 1.0f),
                         gpucast::math::vec4f const& ltr = gpucast::math::vec4f(-1.0f,  1.0f,  1.0f, 1.0f),
                         gpucast::math::vec4f const& rtr = gpucast::math::vec4f( 1.0f,  1.0f,  1.0f, 1.0f) ); // left/right(l/r) - bottom/top(b/t) - front/rear(f/r)

  void set_texcoords   ( gpucast::math::vec4f const& lbf = gpucast::math::vec4f( 0.0f,  0.0f,  0.0f, 1.0f),
                         gpucast::math::vec4f const& rbf = gpucast::math::vec4f( 1.0f,  0.0f,  0.0f, 1.0f),
                         gpucast::math::vec4f const& ltf = gpucast::math::vec4f( 0.0f,  1.0f,  0.0f, 1.0f),
                         gpucast::math::vec4f const& rtf = gpucast::math::vec4f( 1.0f,  1.0f,  0.0f, 1.0f),
                         gpucast::math::vec4f const& lbr = gpucast::math::vec4f( 0.0f,  0.0f,  1.0f, 1.0f),
                         gpucast::math::vec4f const& rbr = gpucast::math::vec4f( 1.0f,  0.0f,  1.0f, 1.0f),
                         gpucast::math::vec4f const& ltr = gpucast::math::vec4f( 0.0f,  1.0f,  1.0f, 1.0f),
                         gpucast::math::vec4f const& rtr = gpucast::math::vec4f( 1.0f,  1.0f,  1.0f, 1.0f) ); // left/right(l/r) - bottom/top(b/t) - front/rear(f/r)

  static std::vector< gpucast::math::vec4f> create_triangle_mesh (  gpucast::math::vec4f const& lbf,
                                                     gpucast::math::vec4f const& rbf,
                                                     gpucast::math::vec4f const& ltf,
                                                     gpucast::math::vec4f const& rtf,
                                                     gpucast::math::vec4f const& lbr,
                                                     gpucast::math::vec4f const& rbr,
                                                     gpucast::math::vec4f const& ltr,
                                                     gpucast::math::vec4f const& rtr );
  
  static void create_triangle_mesh ( gpucast::math::vec4f const& lbf,
                                     gpucast::math::vec4f const& rbf,
                                     gpucast::math::vec4f const& ltf,
                                     gpucast::math::vec4f const& rtf,
                                     gpucast::math::vec4f const& lbr,
                                     gpucast::math::vec4f const& rbr,
                                     gpucast::math::vec4f const& ltr,
                                     gpucast::math::vec4f const& rtr,
                                     std::vector< gpucast::math::vec4f>& attributes,
                                     std::vector<int>&    mesh_indices,
                                     std::vector<int>&    line_indices );
    
  void      set_color       ( float r, float g, float b, float a = 1.0);

private : // methods

  void      _init           ( GLint vertexattrib_index,
                              GLint colorattrib_index,
                              GLint normalattrib_index,
                              GLint texcoordattrib_index );

  void      _init_color     ( float r = 1.0,
                              float g = 1.0,
                              float b = 1.0,
                              float a = 1.0);

  //void      _init_normals   ( );

private : // members

  arraybuffer         _vertices;
  arraybuffer         _colors;
  arraybuffer         _normals;
  arraybuffer         _texcoords;

  elementarraybuffer  _triangles;
  elementarraybuffer  _lines;

  vertexarrayobject   _vao;

};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CUBE_HPP
