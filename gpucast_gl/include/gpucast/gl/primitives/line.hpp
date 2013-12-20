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
#ifndef GPUCAST_GL_LINE_HPP
#define GPUCAST_GL_LINE_HPP

#include <vector>
#include <gpucast/gl/glpp.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/math/vec4.hpp>


namespace gpucast { namespace gl {

  class GPUCAST_GL line : public boost::noncopyable
{
public :

  line                      ( std::vector<vec4f> const& points, 
                              GLint vertexattrib_index = 0, 
                              GLint colorattrib_index = 1, 
                              GLint texcoordattrib_index = 2 );
  ~line                     ( );

public :

  void      set_color       ( std::vector<vec4f> const& color );
  void      set_texcoords   ( std::vector<vec4f> const& texcoords );

  void      attrib_location ( GLint vertexattrib_index, 
                              GLint colorattrib_index, 
                              GLint texcoordattrib_index );

  void      draw            ( float linewidth = 1.0f );

private :

  unsigned            _count;

  arraybuffer         _vertices;
  arraybuffer         _colors;
  arraybuffer         _texcoords;

  vertexarrayobject   _vao;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_LINE_HPP
