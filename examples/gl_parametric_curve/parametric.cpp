/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : parametric.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>

// local includes
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>

#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/math/matrix4x4.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>

#include <gpucast/math/parametric/beziercurve.hpp>
#include <gpucast/math/parametric/point.hpp>




class application
{
public :

  application()
    : _program  (),
      _trackball(new gpucast::gl::trackball),
      _vao      (),
      _attrib0  (),
      _attrib1  ()
  {
    init_shader ();
    init_data   ();

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));

    glEnable(GL_DEPTH_TEST);
  }

  ~application()
  {}

  void init_shader()
  {
    std::string vertexshader_code = R"(
    #version 430 core
      
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 color;    
      
      uniform mat4 modelviewprojectionmatrix; 
      uniform mat4 modelviewmatrix; 
      uniform mat4 normalmatrix; 
      
      out vec4 fragcolor;
      
      void main(void) 
      { 
        fragcolor = color; 
        gl_Position  = modelviewprojectionmatrix * vertex; 
      })";

    std::string fragmentshader_code = R"(
     #version 430 core
      
      in vec4 fragcolor;   
      in vec4 fragposition; 
      
      layout (location = 0) out vec4 color; 
      void main(void) 
      { 
        color = fragcolor; 
      })";

    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);

    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());
    
    vs.compile();
    fs.compile();

    _program.add(&fs);
    _program.add(&vs);

    std::cout << vs.log() << std::endl;
    std::cout << fs.log() << std::endl;

    _program.link();
  }
  
  void init_data()
  {
    gpucast::math::beziercurve<gpucast::math::point<float,6> > bc6d;
    float d0[] = {3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    float d1[] = {1.0f, 1.0f, 1.4f, 1.0f, 1.0f, 0.0f};
    float d2[] = {3.0f, 0.0f, 0.2f, 1.0f, 0.0f, 1.0f};
    float d3[] = {2.0f, 3.0f, 2.5f, 0.0f, 0.0f, 1.0f};

    gpucast::math::point<float, 6> p0(d0);
    gpucast::math::point<float, 6> p1(d1);
    gpucast::math::point<float, 6> p2(d2);
    gpucast::math::point<float, 6> p3(d3);

    bc6d.add(p0);
    bc6d.add(p1);
    bc6d.add(p2);
    bc6d.add(p3);

    std::vector<gpucast::math::vec3f> attrib0; // x,y,z
    std::vector<gpucast::math::vec3f> attrib1; // r,g,b

    unsigned const subdiv = 32;

    for (unsigned i = 0; i != subdiv; ++i) 
    {
      float t = float(i) / (subdiv-1);
      gpucast::math::point<float,6> pt = bc6d.evaluate(t);
      attrib0.push_back(gpucast::math::vec3f(pt[0], pt[1], pt[2]));
      attrib1.push_back(gpucast::math::vec3f(pt[3], pt[4], pt[5]));
    }

    // push also control points after samples
    attrib0.push_back(gpucast::math::vec3f(p0[0], p0[1], p0[2]));
    attrib0.push_back(gpucast::math::vec3f(p1[0], p1[1], p1[2]));
    attrib0.push_back(gpucast::math::vec3f(p2[0], p2[1], p2[2]));
    attrib0.push_back(gpucast::math::vec3f(p3[0], p3[1], p3[2]));

    attrib1.push_back(gpucast::math::vec3f(1.0, 1.0, 1.0));
    attrib1.push_back(gpucast::math::vec3f(1.0, 1.0, 1.0));
    attrib1.push_back(gpucast::math::vec3f(1.0, 1.0, 1.0));
    attrib1.push_back(gpucast::math::vec3f(1.0, 1.0, 1.0));

    // copy data to buffers
    _attrib0.update(attrib0.begin(), attrib0.end());
    _attrib1.update(attrib1.begin(), attrib1.end());

    _vao.bind();

    _vao.attrib_array   (_attrib0, 0, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (0);

    _vao.attrib_array   (_attrib1, 1, 3, GL_FLOAT, false, 0, 0);
    _vao.enable_attrib  (1);

    _vao.unbind();
  }
    

  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv   = view * model;
    gpucast::math::matrix4f mvp  = proj * mv;
    gpucast::math::matrix4f nm   = mv.normalmatrix();

    _program.begin();
  
    _program.set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program.set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program.set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    _vao.bind();
	  glDrawArrays(GL_LINE_STRIP, 0, 32);
    glDrawArrays(GL_POINTS, 32, 4);
    _vao.unbind();

    _program.end();
  }
  

  void run() 
  {
    gpucast::gl::glutwindow::instance().run();
  }


public :

  gpucast::gl::program                       _program;
  std::shared_ptr<gpucast::gl::trackball>  _trackball;

  gpucast::gl::vertexarrayobject             _vao;
  gpucast::gl::arraybuffer                   _attrib0;
  gpucast::gl::arraybuffer                   _attrib1;
};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 0, 0, true);
  glewInit();

  application app;
  app.run();

  return 0;
}
