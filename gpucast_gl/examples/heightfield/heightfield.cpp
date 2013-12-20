/********************************************************************************
*
* Copyright (C) 2009 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : heightfield.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

// local includes
#include <glpp/glut/window.hpp>

#include <glpp/texture2d.hpp>
#include <glpp/program.hpp>
#include <glpp/util/camera.hpp>
#include <glpp/vertexshader.hpp>
#include <glpp/sampler.hpp>
#include <glpp/fragmentshader.hpp>
#include <glpp/primitives/cube.hpp>
#include <glpp/util/trackball.hpp>
#include <glpp/math/matrix4x4.hpp>


class application
{
public :
  enum mode_t {first, second, both};

public :

  application(int argc, char* argv[])
    : _colortex (new glpp::texture2d("../data/map_diffuse.jpg")),
      _bumptex  (new glpp::texture2d("../data/map_height.jpg")),
      _normaltex(new glpp::texture2d("../data/map_normal.jpg")),
      _sampler  (new glpp::sampler),
      _program  (new glpp::program),
      _cube     (new glpp::cube(0, -1, 2, 1)),
      _trackball(new glpp::trackball),
      _camera   (),
      _mode     (both)
  {
    init_shader();
    init_gl();

    glpp::glutwindow::instance().add_eventhandler(_trackball);
    glpp::glutwindow::instance().add_keyevent    ('m', boost::bind(&application::mode, boost::ref(*this), _1, _2));

    _camera.drawcallback(boost::bind(boost::mem_fn(&application::draw), boost::ref(*this)));
    glpp::glutwindow::instance().setcamera(_camera);
  }

  ~application()
  {}

  void init_gl()
  {
    glEnable(GL_DEPTH_TEST);
  }

  void init_shader()
  {
    glpp::vertexshader   vs;
    glpp::fragmentshader fs;

    vs.load("../heightfield.vert");
    fs.load("../heightfield.frag");
    
    vs.compile();
    fs.compile();

    _program->add(&fs);
    _program->add(&vs);

    _program->link();
  }


  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glEnable(glpp::texture2d::target());

    glpp::matrix4f view = glpp::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    glpp::matrix4f model = glpp::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    glpp::matrix4f proj = glpp::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    glpp::matrix4f mv   = view * model;
    glpp::matrix4f mvp  = proj * mv;
    glpp::matrix4f nm   = mv.normalmatrix();

    _program->begin();
  
    _program->set_texture2d("colortex", *_colortex.get(),   0 );
    _program->set_texture2d("bumptex", *_bumptex.get(),     1 );
    _program->set_texture2d("normaltex", *_normaltex.get(), 2 );
    _program->set_uniform4f("lightpos", 0.0f, 0.0f, 0.0f, 1.0f );

    _program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program->set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    switch(_mode) {
      case first :  _program->set_uniform1i("mode", 0); break;
      case second : _program->set_uniform1i("mode", 1); break;
      case both :   _program->set_uniform1i("mode", 2); break;
    };
    _sampler->bind(0);
    _sampler->bind(1);
    _sampler->bind(2);

    _cube->draw();

    _program->end();
  }
  

  void run() 
  {
    glpp::glutwindow::instance().run();
  }


  void mode (int, int)
  {
    switch(_mode) {
      case first :  _mode = second; break;
      case second : _mode = both;   break;
      case both :   _mode = first;  break;
    };
  }


public :

  boost::shared_ptr<glpp::texture2d> _colortex;
  boost::shared_ptr<glpp::texture2d> _bumptex;
  boost::shared_ptr<glpp::texture2d> _normaltex;
  boost::shared_ptr<glpp::sampler>   _sampler;

  boost::shared_ptr<glpp::program>   _program;
  boost::shared_ptr<glpp::cube>      _cube;

  boost::shared_ptr<glpp::trackball> _trackball;
  glpp::camera                       _camera;
  mode_t                             _mode;

};


int main(int argc, char** argv)
{
  glpp::glutwindow::init(argc, argv, 1024, 1024, 10, 10, 3, 3, false);
  glewInit();

  application app(argc, argv);

  app.run();

  return 0;
}


