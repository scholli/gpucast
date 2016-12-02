/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : aabb.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/primitives/obb.hpp"

// header, system
#include <gpucast/gl/primitives/cube.hpp>


namespace gpucast {
  namespace gl {

////////////////////////////////////////////////////////////////////////////////
obb::obb()
: base_type (),
  _cube     (nullptr)
{}


////////////////////////////////////////////////////////////////////////////////
obb::obb( base_type const& b )
: base_type (b),
  _program  (nullptr),
  _cube     (nullptr)
{}


////////////////////////////////////////////////////////////////////////////////
void 
obb::draw(gpucast::math::matrix4x4<float> const& mvp, bool wireframe)
{
  _init ();

  _program->begin();
  {
    _program->set_uniform_matrix4fv( "mvp", 1, 0, &mvp[0]);
    _cube->draw(wireframe);
  }
  _program->end ();
}


////////////////////////////////////////////////////////////////////////////////
void                      
obb::color ( gpucast::math::vec4f const& color )
{
  if (!_cube) {
    _init();
  }

  if (_color != color) {
    _cube->set_color(color[0], color[1], color[2], color[3]);
  }
}

////////////////////////////////////////////////////////////////////////////////
void obb::_init()
{
  _init_program();
  _init_geometry();
}

////////////////////////////////////////////////////////////////////////////////
void
obb::_init_geometry()
{
  if (_cube) {
    return;
  }
  else {
    // create drawable cube
    _cube = std::make_shared<cube>();
    _init_color();

    // create base vertices corresponding to obb
    std::vector<gpucast::math::vec4f> vertices = {
      gpucast::math::vec4f(low()[0],  low()[1],  low()[2], 1.0f),
      gpucast::math::vec4f(high()[0], low()[1],  low()[2], 1.0f),
      gpucast::math::vec4f(low()[0],  high()[1], low()[2], 1.0f),
      gpucast::math::vec4f(high()[0], high()[1], low()[2], 1.0f),
      gpucast::math::vec4f(low()[0],  low()[1],  high()[2], 1.0f),
      gpucast::math::vec4f(high()[0], low()[1],  high()[2], 1.0f),
      gpucast::math::vec4f(low()[0],  high()[1], high()[2], 1.0f),
      gpucast::math::vec4f(high()[0], high()[1], high()[2], 1.0f) };

    // create transformation
    gpucast::math::matrix4f rotation_matrix (orientation().get());
    auto rotation_inverse_matrix = gpucast::math::inverse(rotation_matrix);
    auto model_matrix = gpucast::math::make_translation(center()[0], center()[1], center()[2]) * rotation_matrix;

    // transform OBB's vertices according to translation and orientation
    for (auto& v : vertices) {
      v = model_matrix * v;
    }

    // apply transformed vertices to VAO
    _cube->set_vertices(vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7]);
  }

}

////////////////////////////////////////////////////////////////////////////////
void
obb::_init_program()
{
  if (_program) {
    return;
  }
  else {
    _program = std::make_shared<program>();

    // compile shader and link program
    std::string const vs_src = R"(
    "#version 430 core 
      #extension GL_EXT_gpu_shader4 : enable 
      \n\
      layout (location = 0) in vec4 vertex; \n\
      layout (location = 1) in vec4 color;  \n\
      
      uniform mat4 mvp; 
      
      void main(void) 
      { 
        gl_Position   = mvp * vertex; 
        gl_FrontColor = color; 
      } 
      )";

    gpucast::gl::shader vs(gpucast::gl::vertex_stage);

    vs.set_source(vs_src.c_str());
    vs.compile();

    _program->add(&vs);
    _program->link();
  }
}

////////////////////////////////////////////////////////////////////////////////
void
obb::_init_color()
{
  // create unit cube colors
  std::vector<gpucast::math::vec4f> colors = {
    gpucast::math::vec4f(0.0f, 0.0f, 0.0f, 1.0f),
    gpucast::math::vec4f(1.0f, 0.0f, 0.0f, 1.0f),
    gpucast::math::vec4f(0.0f, 1.0f, 0.0f, 1.0f),
    gpucast::math::vec4f(1.0f, 1.0f, 0.0f, 1.0f),
    gpucast::math::vec4f(0.0f, 0.0f, 1.0f, 1.0f),
    gpucast::math::vec4f(1.0f, 0.0f, 1.0f, 1.0f),
    gpucast::math::vec4f(0.0f, 1.0f, 1.0f, 1.0f),
    gpucast::math::vec4f(1.0f, 1.0f, 1.0f, 1.0f) };

  _cube->set_colors(colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6], colors[7]);
}


  } // namespace gl
} // namespace gpucast 

