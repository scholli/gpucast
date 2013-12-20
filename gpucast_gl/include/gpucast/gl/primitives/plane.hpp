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
#ifndef GPUCAST_GL_PLANE_HPP
#define GPUCAST_GL_PLANE_HPP

#include <gpucast/gl/glpp.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/math/vec4.hpp>


namespace gpucast { namespace gl {

  class GPUCAST_GL plane : public boost::noncopyable
{
public :

  plane                     ( GLint vertexattrib_index, GLint normalattrib_index, GLint texcoordattrib_index );
  ~plane                    ( );

public :

  void      size            ( float width, float height );

  void      texcoords       ( vec4f const& a, 
                              vec4f const& b, 
                              vec4f const& c, 
                              vec4f const& d);

  void      normal          ( vec4f const& n );

  void      attrib_location ( GLint vertexattrib_index, 
                              GLint normalattrib_index, 
                              GLint texcoordattrib_index );

  void      draw            ( );

private :

  arraybuffer         _vertices;
  arraybuffer         _normals;
  arraybuffer         _texcoords;

  vertexarrayobject   _vao;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_PLANE_HPP
