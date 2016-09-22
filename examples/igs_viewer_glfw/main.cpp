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
#include <functional>
#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// local includes
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/util/init_glew.hpp>

#include <gpucast/gl/error.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/gl/primitives/bezierobject.hpp>

#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>


class application : public gpucast::gl::trackball
{
private:

  int                                               _argc;
  char**                                            _argv;

  typedef std::shared_ptr<gpucast::gl::bezierobject> surfaceobject_ptr;
  std::vector<surfaceobject_ptr>                    _objects;

  gpucast::math::bbox3f                             _bbox;

public:

  application(int argc, char** argv)
    : _argc(argc),
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

    auto renderer = gpucast::gl::bezierobject_renderer::instance();
    renderer->recompile();

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
      //filenames.push_back("I:/repositories/gpucast-github/data/vw/scheiben/heckscheibe.igs");
    
    }
    
    for (auto const& file : filenames)
    {
      std::cout << "loading " << file << std::endl;
      auto nurbs_objects = loader.load(file);

      for (auto nurbs_object : nurbs_objects)
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
    }
  }

  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    float near_clip = 0.01f * _bbox.size().abs();
    float far_clip  = 2.0f  * _bbox.size().abs();

    auto renderer = gpucast::gl::bezierobject_renderer::instance();
    renderer->set_nearfar(near_clip, far_clip);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, float(_bbox.size().abs()),
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::vec3f translation = _bbox.center();

    gpucast::math::matrix4f model = gpucast::math::make_translation(shiftx(), shifty(), distance()) * rotation() *
      gpucast::math::make_translation(-translation[0], -translation[1], -translation[2]);

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, near_clip, far_clip);
    gpucast::math::matrix4f mv = view * model;
    gpucast::math::matrix4f mvp = proj * mv;
    gpucast::math::matrix4f nm = mv.normalmatrix();

    renderer->projectionmatrix(proj);
    renderer->modelviewmatrix(mv);

    for (auto const& o : _objects)
    {
      o->draw();
    }

  }

  void resize(int w, int h)
  {
    glViewport(0, 0, w, h);
  }

public:

  virtual void keyboard(unsigned char key, int x, int y) override
  {
    auto renderer = gpucast::gl::bezierobject_renderer::instance();

    // renderer operations
    switch (key)
    {
    case 'c':
    case 'C':
      renderer->recompile();
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
        o->enable_raycasting(!o->enable_raycasting());
        std::cout << "Raycasting set to " << o->enable_raycasting() << std::endl;
        break;
      }
    }
  }
};


std::shared_ptr<application> the_app;


void
glut_display()
{
  the_app->draw();
}

void glfw_mousebutton(GLFWwindow* window, int b, int action, int mods)
{
  glfwMakeContextCurrent(window);

  gpucast::gl::eventhandler::button button;
  gpucast::gl::eventhandler::state state;

  switch (b) {
  case 0: button = gpucast::gl::eventhandler::left; break;
  case 2: button = gpucast::gl::eventhandler::middle; break;
  case 1: button = gpucast::gl::eventhandler::right; break;
  };

  switch (action) {
  case 0: state = gpucast::gl::eventhandler::release; break;
  case 1: state = gpucast::gl::eventhandler::press; break;
  };

  the_app->mouse(button, state, the_app->posx(), the_app->posy());
}

void glfw_motion(GLFWwindow* window, double x, double y)
{
  glfwMakeContextCurrent(window);
  the_app->motion(x, y);
}

void glfw_key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  glfwMakeContextCurrent(window);
  if (action == GLFW_RELEASE)
  {
    switch (key) {
      // ESC key
      case 27:
      {
        the_app.reset();
        std::cout << "reset application" << std::endl;
        exit(0);
      }
      break;
    case 'f':
      //toggle_fullscreen();
      break;
    }
  }

  if (action == GLFW_RELEASE)
  {
    the_app->keyboard(key, the_app->posx(), the_app->posy());
  }
}




int main(int argc, char** argv)
{
  int winx = 1024;
  int winy = 1024;

  glfwInit();

  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

  GLFWwindow* window = glfwCreateWindow(winx, winy, "hello_world_glfw", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  glfwSetKeyCallback(window, glfw_key);
  glfwSetMouseButtonCallback(window, glfw_mousebutton);
  glfwSetCursorPosCallback(window, glfw_motion);

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  glewExperimental = true;
  glewInit();

  the_app = std::make_shared<application>(argc, argv);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
    glut_display();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  the_app.reset();

  return 0;
}