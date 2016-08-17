/********************************************************************************
*
* Copyright (C) Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : atomic_counter.cpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/error.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>

#include <gpucast/math/vec2.hpp>
#include <gpucast/math/vec4.hpp>


class application
{
public :

  application()
  {
    // create graphics resources
    init_shader(); 
    init_vao();

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  } 
    
  void init_shader()  
  {
    std::string vertexshader_code = R"(
     #version 420 core

     layout (location = 0) in vec2 vertex;   
     layout (location = 1) in vec4 texcoord;   

     out vec4 fragtexcoord;  

     void main(void) 
     { 
       fragtexcoord = texcoord; 
       gl_Position  = vec4(vertex.xy, 0.0, 1.0);
     })"; 

    std::string fragmentshader_code = R"(
      #version 420 core

      in vec4 fragtexcoord; 
      
      layout (location = 0) out vec4 color; 
      
      void main(void) 
      { 
        color = fragtexcoord; 
      }
    )";
  
    gpucast::gl::error("0");
    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::error("1");
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);
    gpucast::gl::error("2");
    vs.set_source(vertexshader_code.c_str());
    gpucast::gl::error("3");
    fs.set_source(fragmentshader_code.c_str());
    gpucast::gl::error("4");
    _program.add(&fs);
    gpucast::gl::error("5");
    _program.add(&vs);
    gpucast::gl::error("6");

    std::cout << "vertex shader log : " << vs.log() << std::endl;
    std::cout << "fragment shader log : " << fs.log() << std::endl;

    _program.link();   
  }

  // create vertex array object for a simple triangle with colored (as texture coordinates) attribute
  void init_vao() {

    // declare interleaved vertexarray structure
    struct vertex {
      gpucast::math::vec2f pos;
      gpucast::math::vec4f texcoord;
    };

    // create cpu attribute array
    std::vector<vertex> vertices(3);
    vertices[0] = { gpucast::math::vec2f{ 0.0f, 0.5f }, gpucast::math::vec4f { 1.0f, 0.0f, 0.0f, 1.0f } };
    vertices[1] = { gpucast::math::vec2f{ -0.5f, -0.5f }, gpucast::math::vec4f { 0.0f, 1.0f, 0.0f, 1.0f } };
    vertices[2] = { gpucast::math::vec2f{ 0.5f, -0.5f }, gpucast::math::vec4f { 0.0f, 0.0f, 1.0f, 1.0f } };

    // create simlpe 
    std::vector<unsigned> indices = { 0, 1, 2 };

    // copy attribute and index data into arraybuffers
    _attribute_buffer.bufferdata(vertices.size() * sizeof(vertex), (void*)&vertices[0], GL_STATIC_DRAW);
    _indices.bufferdata(3 * sizeof(unsigned), (void*)&indices[0], GL_STATIC_DRAW);

    // create vertex array object bindings
    _vao.bind();

    // attribute 0 is vertex position
    _vao.attrib_array(_attribute_buffer, 0, 2, GL_FLOAT, false, sizeof(vertex), 0);
    _vao.enable_attrib(0);
    // attribute 1 is texture coordinate
    _vao.attrib_array(_attribute_buffer, 1, 4, GL_FLOAT, false, sizeof(vertex), sizeof(gpucast::math::vec2f));
    _vao.enable_attrib(1);

    _vao.unbind();
  }


  void draw()
  {
    // clear frame buffer
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // bind vertexarrayobject and corresponding element array with triangle indices
    _vao.bind();
    _indices.bind();

    // draw triangle
    _program.begin();
    glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0, 0);
    _program.end();

    _vao.unbind();
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  gpucast::gl::program                     _program;
  gpucast::gl::arraybuffer                 _attribute_buffer;
  gpucast::gl::elementarraybuffer          _indices;
  gpucast::gl::vertexarrayobject           _vao;
};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1920, 1080, 0, 0, 4, 2, true);

  glewExperimental = true;
  glewInit();

  application app;
  app.run();

  return 0;
}
