/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : line.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/line.hpp"

#include <iostream>
#include <vector>
#include <cassert>

#include <gpucast/gl/math/vec4.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
line::line(std::vector<vec4f> const& points,
                 GLint vertexattrib_index,  
                 GLint normalattrib_index, 
                 GLint texcoordattrib_index )
  : _count     ( int(points.size()) ),
    _vao       (),
    _vertices  ( points.size() * sizeof(vec4f) ),
    _colors    ( points.size() * sizeof(vec4f) ),
    _texcoords ( points.size() * sizeof(vec4f) )
{  
  _vertices.buffersubdata (0, unsigned( points.size() * sizeof(vec4f)),  &points.front());
  attrib_location (vertexattrib_index, normalattrib_index, texcoordattrib_index);
}

///////////////////////////////////////////////////////////////////////////////
line::~line()
{}


///////////////////////////////////////////////////////////////////////////////
void      
line::set_color ( std::vector<vec4f> const& color )
{
  assert ( _count == color.size() ); 
  _colors.buffersubdata (0, unsigned( color.size() * sizeof(vec4f)),  &color.front());  
}


///////////////////////////////////////////////////////////////////////////////
void      
line::set_texcoords ( std::vector<vec4f> const& texcoords )
{
  assert ( _count == texcoords.size() ); 
  _colors.buffersubdata (0, unsigned( texcoords.size() * sizeof(vec4f)),  &texcoords.front());  
}


///////////////////////////////////////////////////////////////////////////////
void      
line::attrib_location (GLint vertexattrib_index, GLint colorattrib_index, GLint texcoordattrib_index)
{
  // bind vertices to generic attribute and enable array
  if (vertexattrib_index >= 0) 
  {  
    _vao.attrib_array(_vertices, vertexattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(vertexattrib_index);
  }

  // bind normals to generic attribute and enable array
  if (colorattrib_index >= 0) 
  {
    _vao.attrib_array(_colors, colorattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(colorattrib_index);
  }

  // bind texcoords to generic attribute and enable array
  if (texcoordattrib_index >= 0) 
  {
    _vao.attrib_array(_texcoords, texcoordattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(texcoordattrib_index);
  }
}


///////////////////////////////////////////////////////////////////////////////
void 
line::draw(float linewidth )
{
  glLineWidth(linewidth);
  _vao.bind();
  glDrawArrays(GL_LINE_STRIP, 0, _count);
  _vao.unbind();
}

} } // namespace gpucast / namespace gl
