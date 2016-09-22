/********************************************************************************
*
* Copyright (C) Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : example_ssbo.cpp
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
#include <gpucast/gl/shaderstoragebuffer.hpp>

#include <gpucast/math/vec2.hpp>
#include <gpucast/math/vec4.hpp>

#define SSBO_BINDING_POINT 3

class application
{
public :

  application()
  {
    // create graphics resources
    init_shader(); 
    init_vao();
    init_ssbo();

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  } 
    
  void init_shader()  
  {
    std::string vertexshader_code = R"(
     #version 430 core
     #extension GL_NV_gpu_shader5 : enable
     #extension GL_ARB_shader_storage_buffer_object : require

     layout (location = 0) in vec2 vertex;   
     
     #define SSBO_BINDING_POINT 3
     
     struct ssbo_item
     {
       uint16_t index_0;
       uint16_t Index_1;
       float y;
       vec2 index;
       vec4 color;
     };  

     layout(std430, binding = SSBO_BINDING_POINT) buffer ssbo_name {
       ssbo_item ssbo_data[];
     };

     out vec4 per_vertex_color;  

     void main(void) 
     { 
       per_vertex_color = ssbo_data[gl_VertexID].color; 
       gl_Position  = vec4(vertex.xy, 0.0, 1.0);
     })"; 

    std::string fragmentshader_code = R"(
      #version 430 core

      in vec4 per_vertex_color; 
      
      layout (location = 0) out vec4 color; 
      
      void main(void) 
      { 
        color = per_vertex_color; 
      }
    )";
  
    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);
 
    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());

    _program.add(&fs);
    _program.add(&vs);

    std::cout << "vertex shader log : " << vs.log() << std::endl;
    std::cout << "fragment shader log : " << fs.log() << std::endl;

    _program.link();   
  }

  // create vertex array object for a simple triangle with colored (as texture coordinates) attribute
  void init_vao() {

    // declare interleaved vertexarray structure
    struct vertex {
      gpucast::math::vec2f pos;
    };

    // create cpu attribute array
    std::vector<vertex> vertices(3);
    vertices[0] = { gpucast::math::vec2f{ 0.0f, 0.5f } };
    vertices[1] = { gpucast::math::vec2f{ -0.5f, -0.5f } };
    vertices[2] = { gpucast::math::vec2f{ 0.5f, -0.5f } };

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

    _vao.unbind();
  }

  void init_ssbo() {
    // create ssbo layout and take care of padding!!! new combined-members(vec2 etc.) each 8-byte align!
    struct ssbo_item {
      unsigned short value0;
      unsigned short value1;
      float alpha;
      gpucast::math::vec2f texture_coord;
      gpucast::math::vec4f color;
    };

    // create data
    std::vector<ssbo_item> ssbo_data = {
      { 0U, 4U,  0.0f, gpucast::math::vec2f{ 0.6f, 0.4f }, gpucast::math::vec4f(0.4f, 0.7f, 0.2f, 1.0f) },
      { 1U, 4U,  0.0f, gpucast::math::vec2f{ 0.0f, 0.5f }, gpucast::math::vec4f(0.4f, 0.0f, 0.9f, 1.0f) },
      { 2U, 4U,  0.0f, gpucast::math::vec2f{ 0.5f, 0.0f }, gpucast::math::vec4f(0.7f, 0.5f, 0.0f, 1.0f) }
    };

    _ssbo.bufferdata(ssbo_data.size() * sizeof(ssbo_item), &ssbo_data[0], GL_DYNAMIC_COPY);
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
    _program.set_shaderstoragebuffer("ssbo_name", _ssbo, SSBO_BINDING_POINT);
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
  gpucast::gl::shaderstoragebuffer         _ssbo;
};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1920, 1080, 0, 0, 4, 3, true);

  glewExperimental = true;
  glewInit();

  application app;
  app.run();

  return 0;
}

