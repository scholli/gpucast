/********************************************************************************
*
* Copyright (C) 2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : coordinate_system.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/primitives/coordinate_system.hpp"

#include <iostream>
#include <vector>

#include <gpucast/math/vec4.hpp>
#include <gpucast/gl/error.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
coordinate_system::coordinate_system(GLint vertexattrib_index, GLint colorattrib_index )
  : _vertices ( 6 * sizeof(gpucast::math::vec4f) ),
    _colors   ( 6 * sizeof(gpucast::math::vec4f) ),
    _vao      ()
{  
  // set geometry
  set     ( gpucast::math::vec4f(0.0, 0.0, 0.0, 1.0), 1.0f);
    
  // set colors
  colors  ( gpucast::math::vec4f(1.0, 0.0, 0.0, 1.0), gpucast::math::vec4f(0.0, 1.0, 0.0, 1.0), gpucast::math::vec4f(0.0, 0.0, 1.0, 1.0) );

  // bind attributes to certain locations 
  attrib_location (vertexattrib_index, colorattrib_index);
}

///////////////////////////////////////////////////////////////////////////////
coordinate_system::~coordinate_system()
{}


///////////////////////////////////////////////////////////////////////////////
void      
coordinate_system::set ( gpucast::math::vec4f const& base,
                         float size ) const
{
  std::vector<gpucast::math::vec4f> vertices;

  vertices.push_back(base);
  vertices.push_back(base + gpucast::math::vec4f( size, 0.0f, 0.0f, 0.0f));

  vertices.push_back(base);
  vertices.push_back(base + gpucast::math::vec4f( 0.0f, size, 0.0f, 0.0f));

  vertices.push_back(base);
  vertices.push_back(base + gpucast::math::vec4f( 0.0f, 0.0f, size, 0.0f));

  _vertices.buffersubdata ( 0, unsigned( vertices.size() * sizeof(gpucast::math::vec4f)),  &vertices.front());
}


///////////////////////////////////////////////////////////////////////////////
void      
coordinate_system::colors ( gpucast::math::vec4f const& x_axis,
                            gpucast::math::vec4f const& y_axis,
                            gpucast::math::vec4f const& z_axis ) const
{
  std::vector<gpucast::math::vec4f> colors;

  colors.insert(colors.end(), 2, x_axis);
  colors.insert(colors.end(), 2, y_axis);
  colors.insert(colors.end(), 2, z_axis);

  _colors.buffersubdata(0, unsigned( colors.size() * sizeof(gpucast::math::vec4f)), &colors.front());
}


///////////////////////////////////////////////////////////////////////////////
void      
coordinate_system::attrib_location ( GLint vertexattrib_index, GLint colorattrib_index ) const
{
  // bind vertices to generic attribute and enable array
  if (vertexattrib_index >= 0) 
  {  
    _vao.attrib_array(_vertices, vertexattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(vertexattrib_index);
  }

  // bind colors to generic attribute and enable array
  if (colorattrib_index >= 0) 
  {
    _vao.attrib_array(_colors, colorattrib_index, 4, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib(colorattrib_index);
  }
}


///////////////////////////////////////////////////////////////////////////////
void 
coordinate_system::draw() const
{
  _vao.bind();
  glDrawArrays(GL_LINES, 0, 6);
  _vao.unbind();
}

} } // namespace gpucast / namespace gl
