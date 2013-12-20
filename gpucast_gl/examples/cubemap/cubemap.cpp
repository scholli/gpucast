/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : cubemap.cpp
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
#include <glpp/cubemap.hpp>
#include <glpp/program.hpp>
#include <glpp/util/camera.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/primitives/cube.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/error.hpp>
#include <glpp/math/matrix4x4.hpp>



class application
{
public :

  application()
    : _program  (),
      _cube     (0, -1, 2, 1),
      _trackball(new glpp::trackball),
      _cubemap  (new glpp::cubemap),
      _camera   ()
  {
    init_shader();

    glpp::glutwindow::instance().add_eventhandler(_trackball);

    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);

    glEnable(GL_DEPTH_TEST);

    _cubemap->load( "../panorama_negative_x.jpg", 
                    "../panorama_positive_x.jpg", 
                    "../panorama_negative_y.jpg", 
                    "../panorama_positive_y.jpg", 
                    "../panorama_negative_z.jpg", 
                    "../panorama_positive_z.jpg");

    _cubemap->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _cubemap->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  ~application()
  {}

  void init_shader()
  {
    std::string vertexshader_code =
    "#version 330 compatibility\n \
     #extension GL_ARB_separate_shader_objects : enable \n \
      \n \
      layout (location = 0) in vec4 vertex;   \n \
      layout (location = 1) in vec4 texcoord;   \n \
      layout (location = 2) in vec4 normal;   \n \
      \n \
      uniform mat4 modelviewprojectionmatrix; \n \
      uniform mat4 modelviewmatrix; \n \
      uniform mat4 normalmatrix; \n \
      \n \
      out vec4 fragnormal;  \n \
      out vec4 fragtexcoord;\n \
      out vec4 fragposition;\n \
      \n \
      void main(void) \n \
      { \n \
        fragtexcoord = texcoord; \n \
        fragnormal   = normalmatrix * normal; \n \
        fragposition = modelviewmatrix * vertex; \n \
        gl_Position  = modelviewprojectionmatrix * vertex; \n \
      }\n";

    std::string fragmentshader_code = 
     "#version 330 compatibility\n \
      #extension GL_ARB_separate_shader_objects : enable \n \
      \n \
      in vec4 fragnormal;   \n \
      in vec4 fragtexcoord; \n \
      in vec4 fragposition; \n \
      uniform samplerCube cubemap; \n\
      \n \
      layout (location = 0) out vec4 color; \n \
      \n \
      void main(void) \n \
      { \n \
        vec3 V = normalize(-fragposition.xyz);  \n \
        vec3 N = normalize(fragnormal.xyz); \n \
        float attenuation = min(1.0, 10.0 / length(fragposition.xyz)); \n \
        color = attenuation * vec4(1.0) * dot(N,V) * textureCube(cubemap, reflect(fragposition.xyz, fragnormal.xyz)); \n \
      }\n";

    glpp::vertexshader   vs;
    glpp::fragmentshader fs;

    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());
    
    vs.compile();
    fs.compile();

    _program.add(&fs);
    _program.add(&vs);

    std::cout << "vertex shader log : " << vs.log() << std::endl;
    std::cout << "fragment shader log : " << fs.log() << std::endl;

    _program.link();   
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
    _program.set_cubemap("cubemap", *_cubemap, 0);

    _cube.draw();

    _program.end();
  }
  

  void run() 
  {
    glpp::glutwindow::instance().run();
  }


public :

  glpp::program                       _program;
  glpp::cube                          _cube;
  glpp::camera                        _camera;
  boost::shared_ptr<glpp::trackball>  _trackball;
  boost::shared_ptr<glpp::cubemap>    _cubemap;
};


int main(int argc, char** argv)
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 0, 0, 3, 3, false);
  glewInit();

  application app;
  app.run();

  return 0;
}
