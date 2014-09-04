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
#ifndef GPUCAST_GL_COORDINATE_SYSTEM_HPP
#define GPUCAST_GL_COORDINATE_SYSTEM_HPP

#include <gpucast/gl/glpp.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/math/vec4.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL coordinate_system
{
public :

  coordinate_system           ( GLint vertexattrib_index, GLint colorattrib_index );
  ~coordinate_system          ( );

  coordinate_system(coordinate_system const&  other) = delete;
  coordinate_system operator=(coordinate_system const&  other) = delete;

public :

  void      set               ( gpucast::math::vec4f const&  base,
                                float               size ) const;

  void      colors            ( gpucast::math::vec4f const& x_axis,
                                gpucast::math::vec4f const& y_axis,
                                gpucast::math::vec4f const& z_axis ) const;

  void      attrib_location   ( GLint vertexattrib_index, GLint colorattrib_index ) const;

  void      draw              () const;

private :

  arraybuffer       _vertices;
  arraybuffer       _colors;
  vertexarrayobject _vao;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_COORDINATE_SYSTEM_HPP
