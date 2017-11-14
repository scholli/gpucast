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
#include <gpucast/glut/window.hpp>

#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/program.hpp>

#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/sampler.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/math/matrix4x4.hpp>


class application
{
public :
  enum mode_t {first, second, both};

public :

  application(int argc, char* argv[])
    :
#if 1
      _colortex (new gpucast::gl::texture2d("./data/map_diffuse.jpg")),
      _bumptex  (new gpucast::gl::texture2d("./data/map_height.jpg")),
      _normaltex(new gpucast::gl::texture2d("./data/map_normal.jpg")),
#else
      _colortex (new gpucast::gl::texture2d("./ps_texture_16k.png")),
      _bumptex  (new gpucast::gl::texture2d("./ps_height_16k.png")),
      _normaltex(new gpucast::gl::texture2d("./map_normal.png")),
#endif
      _sampler  (new gpucast::gl::sampler),
      _program  (new gpucast::gl::program),
      _cube     (new gpucast::gl::cube(0, -1, 2, 1)),
      _trackball(new gpucast::gl::trackball),
      _mode     (both)
  {
    init_shader();
    init_gl();

    gpucast::gl::glutwindow::instance().add_eventhandler(_trackball);
    gpucast::gl::glutwindow::instance().add_keyevent    ('m', boost::bind(&application::mode, boost::ref(*this), _1, _2));

    // bind draw loop
    std::function<void()> dcb = std::bind(&application::draw, std::ref(*this));
    gpucast::gl::glutwindow::instance().set_drawfunction(std::make_shared<std::function<void()>>(dcb));
  }

  ~application()
  {}

  void init_gl()
  {
    glEnable(GL_DEPTH_TEST);
  }

  void init_shader()
  {
    gpucast::gl::shader vs(gpucast::gl::vertex_stage);
    gpucast::gl::shader fs(gpucast::gl::fragment_stage);

    vs.load("./heightfield.vert");
    fs.load("./heightfield.frag");
    
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

    glEnable(gpucast::gl::texture2d::target());

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f, 
                                       0.0f, 0.0f, 0.0f, 
                                       0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                           _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv   = view * model;
    gpucast::math::matrix4f mvp  = proj * mv;
    gpucast::math::matrix4f nm   = mv.normalmatrix();

    _program->begin();
  
    _program->set_texture2d("colortex", *_colortex.get(),   0 );
    _program->set_texture2d("bumptex", *_bumptex.get(),     1 );
    _program->set_texture2d("normaltex", *_normaltex.get(), 2 );
    _program->set_uniform4f("lightpos", 0.0f, 0.0f, 0.0f, 1.0f );
    _program->set_uniform1i("texture_height", _colortex->height());
    _program->set_uniform1i("texture_width", _colortex->width());

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
    gpucast::gl::glutwindow::instance().run();
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

  std::shared_ptr<gpucast::gl::texture2d> _colortex;
  std::shared_ptr<gpucast::gl::texture2d> _bumptex;
  std::shared_ptr<gpucast::gl::texture2d> _normaltex;
  std::shared_ptr<gpucast::gl::sampler>   _sampler;

  std::shared_ptr<gpucast::gl::program>   _program;
  std::shared_ptr<gpucast::gl::cube>      _cube;

  std::shared_ptr<gpucast::gl::trackball> _trackball;
  mode_t                             _mode;

};


int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1024, 1024, 10, 10, 0, 0, true);

  glewExperimental = true;
  glewInit();

  application app(argc, argv);

  app.run();

  return 0;
}


