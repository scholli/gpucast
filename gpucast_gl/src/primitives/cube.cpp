/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : cube.hpp
*  project    : gl3pp
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/cube.hpp"

#include <vector>

#include <gpucast/math/vec4.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
cube::cube(int vertexindex, int colorindex, int normalindex, int texcoordindex )
  : _vertices   (),
    _colors     (),
    _normals    (),
    _texcoords  (),
    _triangles  (),
    _lines      (),
    _vao        ()
{
  _init(vertexindex, colorindex, normalindex, texcoordindex);

  //       7______8
  //      /.     /|
  //     / .    / |
  //    3------4  |           +y   +z
  //    |  5. .| ./6           |  /
  //    | .    | /             | /
  //    --------/              ----- +x
  //    1      2

  unsigned mesh_indices[36];
  for (unsigned index = 0; index != 36; ++index) {
    mesh_indices[index] = index;
  }
  _triangles.bufferdata(sizeof(mesh_indices), (void*)mesh_indices, GL_STATIC_DRAW);

  unsigned line_indices[36] = { 0, 2, 1, 2, 3, 4, 4, 5, 
                                6, 8, 7, 8, 9, 10, 10, 11, 
                                12, 14, 16, 17, 20, 21, 22, 23};
  _lines.bufferdata(sizeof(line_indices), (void*)line_indices, GL_STATIC_DRAW);
}


///////////////////////////////////////////////////////////////////////////////
cube::~cube()
{}


///////////////////////////////////////////////////////////////////////////////
void
cube::draw(bool wireframe)
{
  _vao.bind();

  if (wireframe)
  {
    _lines.bind();
    glDrawElementsBaseVertex(GL_LINES, 24, GL_UNSIGNED_INT, 0, 0);
  } else {
    _triangles.bind();
    glDrawElementsBaseVertex(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, 0);
  }

  _vao.unbind();
}


///////////////////////////////////////////////////////////////////////////////
void    
cube::set_vertices (gpucast::math::vec4f const& lbf,// 1
                    gpucast::math::vec4f const& rbf,// 2 
                    gpucast::math::vec4f const& ltf,// 3
                    gpucast::math::vec4f const& rtf,// 4
                    gpucast::math::vec4f const& lbr,// 5
                    gpucast::math::vec4f const& rbr,// 6
                    gpucast::math::vec4f const& ltr,// 7
                    gpucast::math::vec4f const& rtr // 8
                   )
{
  std::vector<gpucast::math::vec4f> vertices = create_triangle_mesh ( lbf, rbf, ltf, rtf, lbr, rbr, ltr, rtr );

  // upload vertices 
  _vertices.buffersubdata ( 0, unsigned( vertices.size() * sizeof(gpucast::math::vec4f)),  &vertices.front());

  // compute normals
  std::vector<gpucast::math::vec3f> normals(36);

  //       6______5
  //      /.     /|
  //     / .    / |
  //    2------1  |           +y   +z
  //    |  7. .| ./8           |  /
  //    | .    | /             | /
  //    --------/              ----- +x
  //    4      3

  gpucast::math::vec3f top    = cross( gpucast::math::vec3f(ltr-ltf), gpucast::math::vec3f(rtr-ltf) );
  gpucast::math::vec3f bottom = cross( gpucast::math::vec3f(rbr-lbf), gpucast::math::vec3f(lbr-lbf) );
  gpucast::math::vec3f front  = cross( gpucast::math::vec3f(ltf-lbf), gpucast::math::vec3f(rbf-lbf) );
  gpucast::math::vec3f back   = cross( gpucast::math::vec3f(rbr-lbr), gpucast::math::vec3f(ltr-lbr) );
  gpucast::math::vec3f left   = cross( gpucast::math::vec3f(lbr-lbf), gpucast::math::vec3f(ltr-lbf) ); 
  gpucast::math::vec3f right  = cross( gpucast::math::vec3f(rtf-rbf), gpucast::math::vec3f(rbr-rbf) ); 

  // use flat shading to get correct per-surface normal
  std::fill_n(normals.begin()     , 6, top   );
  std::fill_n(normals.begin() + 6 , 6, bottom);
  std::fill_n(normals.begin() + 12, 6, front );
  std::fill_n(normals.begin() + 18, 6, back  );
  std::fill_n(normals.begin() + 24, 6, left  );
  std::fill_n(normals.begin() + 30, 6, right );

  _normals.buffersubdata  ( 0, unsigned(  normals.size() * sizeof(gpucast::math::vec3f)),   &normals.front());
}


///////////////////////////////////////////////////////////////////////////////
void
cube::set_color ( float r, float g, float b, float a)
{
  _init_color(r, g, b, a);
}


///////////////////////////////////////////////////////////////////////////////
void
cube::_init ( GLint vertexattrib_index,
              GLint colorattrib_index,
              GLint normalattrib_index,
              GLint texcoordattrib_index)
{
  // allocate memory
  _vertices.bufferdata  (36 * sizeof(gpucast::math::vec4f), 0);
  _normals.bufferdata   (36 * sizeof(gpucast::math::vec3f), 0);
  _colors.bufferdata    (36 * sizeof(gpucast::math::vec4f), 0);
  _texcoords.bufferdata (36 * sizeof(gpucast::math::vec4f), 0);

  // set data for cube
  set_vertices();
  set_texcoords();
  _init_color();

  if (vertexattrib_index >=0) {
    // bind vertices to generic attribute and enable array
    _vao.attrib_array(_vertices, vertexattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(vertexattrib_index);
  }

  if (colorattrib_index >=0) {
    // bind vertices to generic attribute and enable array
    _vao.attrib_array(_colors, colorattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(colorattrib_index);
  }

  if (normalattrib_index >=0) {
    // bind vertices to generic attribute and enable array
    _vao.attrib_array(_normals, normalattrib_index, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(normalattrib_index);
  }

  if (texcoordattrib_index >=0) {
    // bind vertices to generic attribute and enable array
    _vao.attrib_array(_texcoords, texcoordattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(texcoordattrib_index);
  }
}


///////////////////////////////////////////////////////////////////////////////
void
cube::set_texcoords (gpucast::math::vec4f const& lbf,
                     gpucast::math::vec4f const& rbf,
                     gpucast::math::vec4f const& ltf,
                     gpucast::math::vec4f const& rtf,
                     gpucast::math::vec4f const& lbr,
                     gpucast::math::vec4f const& rbr,
                     gpucast::math::vec4f const& ltr,
                     gpucast::math::vec4f const& rtr )

{
  std::vector<gpucast::math::vec4f> texcoords = create_triangle_mesh ( lbf, rbf, ltf, rtf, lbr, rbr, ltr, rtr );
  _texcoords.buffersubdata( 0, unsigned(texcoords.size() * sizeof(gpucast::math::vec4f)), &texcoords.front());
}


///////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::math::vec4f>
cube::create_triangle_mesh ( gpucast::math::vec4f const& lbf,
                             gpucast::math::vec4f const& rbf,
                             gpucast::math::vec4f const& ltf,
                             gpucast::math::vec4f const& rtf,
                             gpucast::math::vec4f const& lbr,
                             gpucast::math::vec4f const& rbr,
                             gpucast::math::vec4f const& ltr,
                             gpucast::math::vec4f const& rtr )
{
  std::vector<gpucast::math::vec4f> attrib_array;

  //       7______8
  //      /.     /|
  //     / .    / |
  //    3------4  |           +y   +z
  //    |  5. .| ./6           |  /
  //    | .    | /             | /
  //    --------/              ----- +x
  //    1      2

  // top
  attrib_array.push_back(ltf); // 3
  attrib_array.push_back(rtr); // 8
  attrib_array.push_back(ltr); // 7 
  attrib_array.push_back(ltf); // 3 
  attrib_array.push_back(rtf); // 4 
  attrib_array.push_back(rtr); // 8 
      
  // bottom                 
  attrib_array.push_back(lbr); // 5 
  attrib_array.push_back(rbf); // 2 
  attrib_array.push_back(lbf); // 1 
  attrib_array.push_back(lbr); // 5 
  attrib_array.push_back(rbr); // 6 
  attrib_array.push_back(rbf); // 2 
                   
  // front                  
  attrib_array.push_back(lbf); // 1 
  attrib_array.push_back(rtf); // 4 
  attrib_array.push_back(ltf); // 3 
  attrib_array.push_back(lbf); // 1 
  attrib_array.push_back(rbf); // 2 
  attrib_array.push_back(rtf); // 4 
  
  // back                   
  attrib_array.push_back(rbr); // 6 
  attrib_array.push_back(ltr); // 7 
  attrib_array.push_back(rtr); // 8 
  attrib_array.push_back(rbr); // 6 
  attrib_array.push_back(lbr); // 5 
  attrib_array.push_back(ltr); // 7 
                            
  // left                   
  attrib_array.push_back(lbr); // 5 
  attrib_array.push_back(ltf); // 3 
  attrib_array.push_back(ltr); // 7 
  attrib_array.push_back(lbr); // 5 
  attrib_array.push_back(lbf); // 1 
  attrib_array.push_back(ltf); // 3 
               
  // right                  
  attrib_array.push_back(rbf); // 2 
  attrib_array.push_back(rtr); // 8 
  attrib_array.push_back(rtf); // 4 
  attrib_array.push_back(rbf); // 2 
  attrib_array.push_back(rbr); // 6 
  attrib_array.push_back(rtr); // 8 

  return attrib_array;
}


///////////////////////////////////////////////////////////////////////////////
void
cube::_init_color(float r, float g, float b, float a)
{
  std::vector<gpucast::math::vec4f> colors(36);

  std::fill(colors.begin(), colors.end(), gpucast::math::vec4f(r, g, b, a));

  _colors.buffersubdata( 0, unsigned( colors.size() * sizeof(gpucast::math::vec4f)), &colors.front());
}


} } // namespace gpucast / namespace gl
