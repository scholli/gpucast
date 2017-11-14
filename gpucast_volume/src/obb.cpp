/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : obb.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/obb.hpp"

// header, system
#include <gpucast/math/vec3.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>


namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
obb::obb ( )
  : base_type     (),
    _initialized  (false),
    _color        ( 0.5f, 0.5f, 0.5f, 1.0f )
{}


////////////////////////////////////////////////////////////////////////////////
obb::obb ( base_type const& b )
  : base_type    ( b ),
    _initialized (false),
    _color       ( 0.5f, 0.5f, 0.5f, 1.0f )
{}


////////////////////////////////////////////////////////////////////////////////
obb::~obb ()
{}


////////////////////////////////////////////////////////////////////////////////
void
obb::draw ( gpucast::math::matrix4x4<float> const& mvp )
{
  _init ();

  _program->begin();
  {
    _program->set_uniform_matrix4fv( "mvp", 1, 0, &mvp[0]);
    _arrayobject->bind();
    {
      glDrawArrays ( GL_LINES, 0, 24 );
    }
    _arrayobject->unbind();
  }
  _program->end ();
}


////////////////////////////////////////////////////////////////////////////////
void
obb::color ( gpucast::math::vec4f const& color )
{
  if ( color != _color )
  {
    _color = color;

    // if already uploaded to GPU -> update color buffer
    if ( _initialized )
    {
      std::vector<gpucast::math::vec4f> colors (8);
      std::fill ( colors.begin(), colors.end(), gpucast::math::vec4f(_color) );
      _colorarray->update ( colors.begin(), colors.end() );
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
void obb::_init ( )
{
  // if already initialized -> initialize
  if (_initialized) return;

  // allocate ressources
  _vertexarray  = std::shared_ptr<gpucast::gl::arraybuffer>        ( new gpucast::gl::arraybuffer );
  _colorarray   = std::shared_ptr<gpucast::gl::arraybuffer>        ( new gpucast::gl::arraybuffer );
  _arrayobject  = std::shared_ptr<gpucast::gl::vertexarrayobject>  ( new gpucast::gl::vertexarrayobject );
  _program      = std::shared_ptr<gpucast::gl::program>            ( new gpucast::gl::program );

  // compile shader and link program
  std::string const vs_src =
    "#version 330 compatibility \n \
     #extension GL_EXT_gpu_shader4 : enable \n \
      \n\
      layout (location = 0) in vec4 vertex; \n\
      layout (location = 1) in vec4 color;  \n\
      \n \
      uniform mat4 mvp; \n \
      \n \
      void main(void) \n \
      { \n \
        gl_Position   = mvp * vertex; \n \
        gl_FrontColor = color; \n \
      } \n \
      ";

  gpucast::gl::shader vs(gpucast::gl::vertex_stage);;

  vs.set_source(vs_src.c_str());
  vs.compile();

  _program->add  (&vs);
  _program->link ();

  // generate data and copy to GPU
  std::vector<gpucast::math::vec3f> corners;
  generate_corners ( std::back_inserter( corners ) );

  std::vector<gpucast::math::vec3f> vertices;

  // points whose index differ one bit form one edge
  for (unsigned i = 0; i != corners.size(); ++i)
  {
    if ( i & 0x01 ) {
      unsigned k = i ^ 0x01;
      vertices.push_back(corners[i]);
      vertices.push_back(corners[k]);
    }

    if ( i & 0x02 ) {
      unsigned k = i ^ 0x02;
      vertices.push_back(corners[i]);
      vertices.push_back(corners[k]);
    }

    if ( i & 0x04 ) {
      unsigned k = i ^ 0x04;
      vertices.push_back(corners[i]);
      vertices.push_back(corners[k]);
    }
  }

  std::vector<gpucast::math::vec4f> colors ( vertices.size() );
  std::fill(colors.begin(), colors.end(), gpucast::math::vec4f(_color));

  _vertexarray->update ( vertices.begin(),  vertices.end() );
  _colorarray->update  ( colors.begin(),    colors.end()   );

  _arrayobject->bind();

  _arrayobject->attrib_array  ( *_vertexarray, 0, 3, GL_FLOAT, false, 0, 0 );
  _arrayobject->enable_attrib ( 0 );
  _arrayobject->attrib_array  ( *_colorarray, 1, 4, GL_FLOAT, false, 0, 0 );
  _arrayobject->enable_attrib ( 1 );

  _arrayobject->unbind();

  _initialized = true;
}



} // namespace gpucast
