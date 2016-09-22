/********************************************************************************
*
* Copyright (C) Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : transform_feedback.cpp
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
#include <gpucast/gl/transformfeedback.hpp>
#include <gpucast/gl/transformfeedback_query.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>

#include <gpucast/math/vec2.hpp>
#include <gpucast/math/vec4.hpp>

// use this to use vertex shader only, otherwise tranform feedback is generated with geometry shader
#define VERTEX_SHADER_ONLY 0

// declare interleaved vertexarray structure
struct vertex {
  gpucast::math::vec2f pos;
  gpucast::math::vec4f texcoord;
};

class application
{
public :

  application()
  {
    // create graphics resources
    init_shader(); 
    init_vao();
    init_xfb();

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  } 
    
  void init_xfb() 
  {
    // initialize ressources
    _xfb.feedback = std::make_shared<gpucast::gl::transform_feedback>();
    _xfb.vertex_array_object = std::make_shared<gpucast::gl::vertexarrayobject>();
    _xfb.buffer = std::make_shared<gpucast::gl::arraybuffer>(1024, GL_STATIC_DRAW);

    // create layout for interleaved transform feedback array buffer object
    _xfb.vertex_array_object->bind();
    {
      _xfb.vertex_array_object->attrib_array(*_xfb.buffer, 0, 2, GL_FLOAT, false, sizeof(vertex), 0);
      _xfb.vertex_array_object->attrib_array(*_xfb.buffer, 1, 4, GL_FLOAT, false, sizeof(vertex), sizeof(gpucast::math::vec2f));

      _xfb.vertex_array_object->enable_attrib(0);
      _xfb.vertex_array_object->enable_attrib(1);
    }
    _xfb.vertex_array_object->unbind();

    // bind buffer to bind array buffer as target to transform feedback 
    _xfb.feedback->bind();
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, _xfb.buffer->id());
    _xfb.feedback->unbind();
  }

  void init_shader()  
  {
#if VERTEX_SHADER_ONLY
    // create xfb program - xfb decreases size of triangle
    std::string xfb_vs_code = R"(
     #version 420 core
     layout (location = 0) in vec2 vertex;   
     layout (location = 1) in vec4 texcoord;   

     out vec2 out_pos;
     out vec4 out_texcoord;

     void main(void) 
     { 
       out_pos      = vertex.xy / 2.0;
       out_texcoord = texcoord;
     })";

    gpucast::gl::shader xfb_vs(gpucast::gl::vertex_stage);
    xfb_vs.set_source(xfb_vs_code.c_str());
    _xfb_program.add(&xfb_vs);
#else
    // create xfb program - xfb decreases size of triangle
    std::string xfb_vs_code = R"(
     #version 420 core
     layout (location = 0) in vec2 vertex;   
     layout (location = 1) in vec4 texcoord;   

     out vec2 varying_pos;
     out vec4 varying_texcoord;

     void main(void) 
     { 
       varying_pos      = vertex.xy;
       varying_texcoord = texcoord;
     })";

    std::string xfb_gs_code = R"(
    #version 420 core
    #extension GL_ARB_enhanced_layouts : enable

    layout(triangles) in; 
    layout(triangle_strip, max_vertices = 6) out;

    in vec2 varying_pos[3];
    in vec4 varying_texcoord[3];

    layout (xfb_offset=0)  out vec2 out_pos;
    layout (xfb_offset=8) out vec4 out_texcoord;

    void main(void)
    {
      for (int i=0; i!= 3; ++i) {
        out_pos = varying_pos[i].xy / 2.0 + vec2(0.5, 0.0);
        out_texcoord = varying_texcoord[i];
        EmitVertex();
      }
      EndPrimitive(); 

      for (int i=0; i!= 3; ++i) {
        out_pos = varying_pos[i].xy / 2.0 - vec2(0.5, 0.0);;
        out_texcoord = varying_texcoord[i];
        EmitVertex();
      }
      EndPrimitive(); 
    })";

    gpucast::gl::shader xfb_vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader xfb_gs(gpucast::gl::geometry_stage);

    xfb_vs.set_source(xfb_vs_code.c_str());
    xfb_gs.set_source(xfb_gs_code.c_str());

    _xfb_program.add(&xfb_vs);
    _xfb_program.add(&xfb_gs);
#endif

    // bind output of xfb program
    GLchar const * varyings[] = { "out_pos", "out_texcoord" };
    glTransformFeedbackVaryings(_xfb_program.id(), 2, varyings, GL_INTERLEAVED_ATTRIBS);
    _xfb_program.link();

    // create final draw program
    std::string draw_vs = R"(
     #version 420 core

     layout (location = 0) in vec2 vertex;   
     layout (location = 1) in vec4 texcoord;   

     out vec4 fragtexcoord;  

     void main(void) { 
       gl_Position  = vec4(vertex.xy, 0.0, 1.0);
       fragtexcoord = texcoord; 
     })"; 

    std::string draw_fs = R"(
      #version 420 core

      in vec4 fragtexcoord; 
      
      layout (location = 0) out vec4 color; 
      
      void main(void) { 
        color = fragtexcoord; 
      }
    )";
  

    gpucast::gl::shader vs(gpucast::gl::vertex_stage); 
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);

    vs.set_source(draw_vs.c_str());
    fs.set_source(draw_fs.c_str());

    _program.add(&fs);
    _program.add(&vs);

    _program.link();   
  }

  // create vertex array object for a simple triangle with colored (as texture coordinates) attribute
  void init_vao() {

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
    {
      // attribute 0 is vertex position
      _vao.attrib_array(_attribute_buffer, 0, 2, GL_FLOAT, false, sizeof(vertex), 0);
      _vao.enable_attrib(0);
      // attribute 1 is texture coordinate
      _vao.attrib_array(_attribute_buffer, 1, 4, GL_FLOAT, false, sizeof(vertex), sizeof(gpucast::math::vec2f));
      _vao.enable_attrib(1);
    }
    _vao.unbind();
  }


  void draw()
  {
    // clear frame buffer
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    ///////////////////////////////////////////////////////////////////////////
    // 1. pass : generate transform feedback
    ///////////////////////////////////////////////////////////////////////////
    // disable rasterizastion
    glEnable(GL_RASTERIZER_DISCARD);
    
    // draw triangle
    _xfb_program.begin();
    {
      // bind vertexarrayobject and corresponding element array with triangle indices
      _vao.bind();
      {
        _indices.bind();
        _query.begin();
        
        _xfb.feedback->bind();
        _xfb.feedback->begin(GL_TRIANGLES);
        glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0, 0);
        _xfb.feedback->end();
        _xfb.feedback->unbind();
        _indices.unbind();
        _query.end();
      }
      _vao.unbind();
    }
    _xfb_program.end();
    glDisable(GL_RASTERIZER_DISCARD);

    std::cout << "Primitives written : " << _query.primitives_written() << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // 2. pass : draw transform feedback
    ///////////////////////////////////////////////////////////////////////////
    _program.begin();
    {
      _xfb.vertex_array_object->bind();
      glDrawTransformFeedback(GL_TRIANGLES, _xfb.feedback->id());
    }
    _program.end();
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  gpucast::gl::program                     _program;
  gpucast::gl::program                     _xfb_program;
  gpucast::gl::arraybuffer                 _attribute_buffer;
  gpucast::gl::elementarraybuffer          _indices;
  gpucast::gl::vertexarrayobject           _vao;
  gpucast::gl::transform_feedback_buffer   _xfb;
  gpucast::gl::transformfeedback_query     _query;
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
