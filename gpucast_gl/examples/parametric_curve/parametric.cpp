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
#include <glpp/glut/window.hpp>

#include <glpp/program.hpp>
#include <glpp/util/camera.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/primitives/cube.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/math/matrix4x4.hpp>

#include <glpp/arraybuffer.hpp>
#include <glpp/error.hpp>
#include <glpp/vertexarrayobject.hpp>

#include <tml/parametric/beziercurve.hpp>
#include <tml/parametric/point.hpp>




class application
{
public :

  application()
    : _program  (),
      _trackball(new glpp::trackball),
      _camera   (),
      _vao      (),
      _attrib0  (),
      _attrib1  ()
  {
    init_shader ();
    init_data   ();

    glpp::glutwindow::instance().add_eventhandler(_trackball);

    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);

    glEnable(GL_DEPTH_TEST);
  }

  ~application()
  {}

  void init_shader()
  {
    std::string vertexshader_code =
    "#version 330 compatibility\n \
      \n \
      layout (location = 0) in vec4 vertex;   \n \
      layout (location = 1) in vec4 color;    \n \
      \n \
      uniform mat4 modelviewprojectionmatrix; \n \
      uniform mat4 modelviewmatrix; \n \
      uniform mat4 normalmatrix; \n \
      \n \
      out vec4 fragcolor;\n \
      \n \
      void main(void) \n \
      { \n \
        fragcolor = color; \n \
        gl_Position  = modelviewprojectionmatrix * vertex; \n \
      }\n";

    std::string fragmentshader_code = 
     "#version 330 compatibility\n \
      \n \
      in vec4 fragcolor;   \n \
      in vec4 fragposition; \n \
      \n \
      layout (location = 0) out vec4 color; \n \
      void main(void) \n \
      { \n \
        color = fragcolor; \n \
      }\n";

    glpp::vertexshader   vs;
    glpp::fragmentshader fs;

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
    tml::beziercurve<tml::point<float,6> > bc6d;
    float d0[] = {3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    float d1[] = {1.0f, 1.0f, 1.4f, 1.0f, 1.0f, 0.0f};
    float d2[] = {3.0f, 0.0f, 0.2f, 1.0f, 0.0f, 1.0f};
    float d3[] = {2.0f, 3.0f, 2.5f, 0.0f, 0.0f, 1.0f};

    tml::point<float, 6> p0(d0);
    tml::point<float, 6> p1(d1);
    tml::point<float, 6> p2(d2);
    tml::point<float, 6> p3(d3);

    bc6d.add(p0);
    bc6d.add(p1);
    bc6d.add(p2);
    bc6d.add(p3);

    std::vector<glpp::vec3f> attrib0; // x,y,z
    std::vector<glpp::vec3f> attrib1; // r,g,b

    unsigned const subdiv = 32;

    for (unsigned i = 0; i != subdiv; ++i) 
    {
      float t = float(i) / (subdiv-1);
      tml::point<float,6> pt = bc6d.evaluate(t);
      attrib0.push_back(glpp::vec3f(pt[0], pt[1], pt[2]));
      attrib1.push_back(glpp::vec3f(pt[3], pt[4], pt[5]));
    }

    // push also control points after samples
    attrib0.push_back(glpp::vec3f(p0[0], p0[1], p0[2]));
    attrib0.push_back(glpp::vec3f(p1[0], p1[1], p1[2]));
    attrib0.push_back(glpp::vec3f(p2[0], p2[1], p2[2]));
    attrib0.push_back(glpp::vec3f(p3[0], p3[1], p3[2]));

    attrib1.push_back(glpp::vec3f(1.0, 1.0, 1.0));
    attrib1.push_back(glpp::vec3f(1.0, 1.0, 1.0));
    attrib1.push_back(glpp::vec3f(1.0, 1.0, 1.0));
    attrib1.push_back(glpp::vec3f(1.0, 1.0, 1.0));

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

    glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    glpp::matrix4f model = glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    glpp::matrix4f proj = glpp::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    glpp::matrix4f mv   = view * model;
    glpp::matrix4f mvp  = proj * mv;
    glpp::matrix4f nm   = mv.normalmatrix();

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
    glpp::glutwindow::instance().run();
  }


public :

  glpp::program                       _program;
  glpp::camera                        _camera;
  boost::shared_ptr<glpp::trackball>  _trackball;

  glpp::vertexarrayobject             _vao;
  glpp::arraybuffer                   _attrib0;
  glpp::arraybuffer                   _attrib1;
};


int main(int argc, char** argv)
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 3, 3, false);
  glewInit();

  application app;
  app.run();

  return 0;
}
