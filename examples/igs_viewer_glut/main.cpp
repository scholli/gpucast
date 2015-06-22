/********************************************************************************
*
* Copyright (C) Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : simple.cpp
*  description:
*
********************************************************************************/

// system includes
#include <iostream>
#include <map>

// local includes
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/util/init_glew.hpp>

#include <gpucast/gl/error.hpp>
#include <gpucast/gl/timer_query.hpp>
#include <gpucast/glut/window.hpp>
#include <gpucast/math/matrix4x4.hpp> 
#include <gpucast/gl/primitives/bezierobject.hpp>

#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>


class application : public gpucast::gl::trackball
{
private : 

  int                                               _argc;
  char**                                            _argv;

  typedef std::shared_ptr<gpucast::gl::bezierobject> surfaceobject_ptr;
  std::vector<surfaceobject_ptr>                    _objects;
  gpucast::gl::timer_query                          _query;
  gpucast::math::bbox3f                             _bbox;

public:
  
  application(int argc, char** argv)
    :  _argc(argc),
       _argv(argv)
  {
    initialize();
  }

  ~application()
  {}

  void initialize()
  {
    glEnable(GL_DEPTH_TEST);

    gpucast::igs_loader loader;
    gpucast::surface_converter converter;

    auto& renderer = gpucast::gl::bezierobject_renderer::instance();

    renderer.add_search_path("../../../../");
    renderer.add_search_path("../../../");
    renderer.add_search_path("../../");
    renderer.add_search_path("../");
    renderer.recompile();

    bool initialized_bbox = false;
    std::vector<std::string> filenames;

    if (_argc > 1)
    {
      for (int i = 1; i < _argc; ++i)
      {
        filenames.push_back(_argv[i]);
      }
    }
    else {
      filenames.push_back("data/part.igs");
    }

    for (auto const& file : filenames)
    {
      std::cout << "loading " << file << std::endl;
      auto nurbs_object = loader.load(file);

      if (nurbs_object)
      {
        auto bezier_object = std::make_shared<gpucast::beziersurfaceobject>();
        converter.convert(nurbs_object, bezier_object);

        if (!initialized_bbox) {
          _bbox.min = bezier_object->bbox().min;
          _bbox.max = bezier_object->bbox().max;
        }
        else {
          _bbox.merge(bezier_object->bbox().min);
          _bbox.merge(bezier_object->bbox().max);
        }

        bezier_object->init();

        gpucast::gl::material mat;
        mat.randomize(0.05f, 1.0f, 0.1f, 20.0f, 1.0f);

        auto drawable = std::make_shared<gpucast::gl::bezierobject>(*bezier_object);
        drawable->set_material(mat);

        _objects.push_back(drawable);
      }
      else {
        std::cerr << "failed to load " << file << std::endl;
        std::cerr << loader.error_message() << std::endl;
      }
    }
  }


  void draw()
  {
    _query.begin();

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    auto& renderer = gpucast::gl::bezierobject_renderer::instance();
    renderer.set_nearfar(0.01f * _bbox.size().abs(), 1.5f  * _bbox.size().abs());

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, float(_bbox.size().abs()),
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::vec3f translation = _bbox.center();

    gpucast::math::matrix4f model = gpucast::math::make_translation(shiftx(), shifty(), distance()) * rotation() *
                                  gpucast::math::make_translation(-translation[0], -translation[1], -translation[2]);

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv = view * model;
    gpucast::math::matrix4f mvp = proj * mv;
    gpucast::math::matrix4f nm = mv.normalmatrix();

    renderer.projectionmatrix(proj);
    renderer.modelviewmatrix(mv);

    for (auto const& o : _objects) {
      o->draw();
    }

    _query.end();

    std::cout << "fps: " << double(1000) / _query.result_wait() << "\r";
  }

  void resize(int w, int h)
  {
    glViewport(0, 0, w, h);
  }

public:

  virtual void keyboard(unsigned char key, int x, int y) override
  {
    auto& renderer = gpucast::gl::bezierobject_renderer::instance();

    // renderer operations
    switch (key)
    {
    case 'c':
      renderer.recompile();
      std::cout << "Shaders recompiled" << std::endl;
      break;
    }

    // per object operations
    for (auto const& o : _objects)
    {
      switch (key)
      {
      case 'b':
        o->culling(!o->culling());
        std::cout << "Backface culling set to " << o->culling() << std::endl;
        break;
      case 'i':
        o->max_newton_iterations(std::max(1U, o->max_newton_iterations() - 1));
        std::cout << "Newton iterations set to " << o->max_newton_iterations() << std::endl;
        break;
      case 'I':
        o->max_newton_iterations(o->max_newton_iterations() + 1);
        std::cout << "Newton iterations set to " << o->max_newton_iterations() << std::endl;
        break;
      case 'r':
        o->raycasting(!o->raycasting());
        std::cout << "Raycasting set to " << o->raycasting() << std::endl;
        break;
      }
    }
  }
};

int main(int argc, char** argv)
{
  int winx = 1024;
  int winy = 1024;

  gpucast::gl::glutwindow::init(argc, argv, winx, winy, 100, 100, 4, 3, true);
  auto& win = gpucast::gl::glutwindow::instance();

  glewExperimental = true;
  gpucast::gl::init_glew(std::cout);

  auto the_app = std::make_shared<application>(argc, argv);

  win.add_eventhandler(the_app);
  
  auto draw_fun = std::make_shared<std::function<void()>>(std::bind(&application::draw, the_app));
  win.set_drawfunction(draw_fun);

  win.run();

  return 0;
}
