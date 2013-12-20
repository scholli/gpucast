/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : plane.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/plane.hpp"

#include <iostream>
#include <vector>

#include <gpucast/gl/math/vec4.hpp>
#include <gpucast/gl/error.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
plane::plane(GLint vertexattrib_index, GLint normalattrib_index, GLint texcoordattrib_index)
  : _vao      (),
    _vertices ( 4 * sizeof(vec4f) ),
    _normals  ( 4 * sizeof(vec4f) ),
    _texcoords( 4 * sizeof(vec4f) )
{  
  size      ( 2.0f, 2.0f );

  texcoords ( vec4f(0.0, 0.0, 0.0, 0.0),
              vec4f(1.0, 0.0, 0.0, 0.0),
              vec4f(1.0, 1.0, 0.0, 0.0),
              vec4f(0.0, 1.0, 0.0, 0.0));

  normal ( vec4f(0.0, 0.0, 1.0, 0.0));
  attrib_location (vertexattrib_index, normalattrib_index, texcoordattrib_index);
}

///////////////////////////////////////////////////////////////////////////////
plane::~plane()
{}

///////////////////////////////////////////////////////////////////////////////
void      
plane::size ( float width, float height )
{
  // xy-plane
  std::vector<vec4f> vertices;
  vertices.push_back(vec4f( -width/2.0f, -height/2.0f,  0.0, 1.0));
  vertices.push_back(vec4f(  width/2.0f, -height/2.0f,  0.0, 1.0));
  vertices.push_back(vec4f(  width/2.0f,  height/2.0f,  0.0, 1.0));
  vertices.push_back(vec4f( -width/2.0f,  height/2.0f,  0.0, 1.0));

  _vertices.buffersubdata (0, unsigned( vertices.size() * sizeof(vec4f)),  &vertices.front());
}

///////////////////////////////////////////////////////////////////////////////
void      
plane::texcoords (vec4f const& a, 
                  vec4f const& b, 
                  vec4f const& c, 
                  vec4f const& d)
{
  std::vector<vec4f> tmp(4);

  tmp.at(0) = a;
  tmp.at(1) = b;
  tmp.at(2) = c;
  tmp.at(3) = d;
  
  _texcoords.buffersubdata (0, tmp.size() * sizeof(vec4f), &tmp.front());
}


///////////////////////////////////////////////////////////////////////////////
void      
plane::normal (vec4f const& n)
{
  // normal in z-axis
  std::vector<vec4f> normals(4, n);
  _normals.buffersubdata (0, unsigned(  normals.size() * sizeof(vec4f)),   &normals.front());
}


///////////////////////////////////////////////////////////////////////////////
void      
plane::attrib_location (GLint vertexattrib_index, GLint normalattrib_index, GLint texcoordattrib_index)
{
  // bind vertices to generic attribute and enable array
  if (vertexattrib_index >= 0) 
  {  
    _vao.attrib_array(_vertices, vertexattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(vertexattrib_index);
  }

  // bind normals to generic attribute and enable array
  if (normalattrib_index >= 0) 
  {
    _vao.attrib_array(_normals, normalattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(normalattrib_index);
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
plane::draw()
{
  _vao.bind();
  glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
  _vao.unbind();
}

} } // namespace gpucast / namespace gl
