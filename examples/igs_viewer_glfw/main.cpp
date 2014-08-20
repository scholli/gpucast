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
#include <gpucast/gl/math/matrix4x4.hpp>
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

    auto& renderer = gpucast::gl::bezierobject_renderer::instance();

    renderer.add_search_path("../../../../");
    renderer.add_search_path("../../../");
    renderer.add_search_path("../../");
    renderer.add_search_path("../");
    renderer.recompile();

    bool initialized_bbox = false;
    for (int i = 1; i < _argc; ++i)
    {
      std::cout << "loading " << _argv[i] << std::endl;
      auto nurbs_object = loader.load(_argv[i]);

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
        std::cerr << "failed to load " << _argv[i] << std::endl;
        std::cerr << loader.error_message() << std::endl;
      }
    }
  }

  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    auto& renderer = gpucast::gl::bezierobject_renderer::instance();
    renderer.set_nearfar(0.01f * _bbox.size().abs(), 1.5f  * _bbox.size().abs());

    gpucast::gl::matrix4f view = gpucast::gl::lookat(0.0f, 0.0f, float(_bbox.size().abs()),
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::gl::vec3f translation = _bbox.center();

    gpucast::gl::matrix4f model = gpucast::gl::make_translation(shiftx(), shifty(), distance()) * rotation() *
      gpucast::gl::make_translation(-translation[0], -translation[1], -translation[2]);

    gpucast::gl::matrix4f proj = gpucast::gl::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::gl::matrix4f mv = view * model;
    gpucast::gl::matrix4f mvp = proj * mv;
    gpucast::gl::matrix4f nm = mv.normalmatrix();

    renderer.projectionmatrix(proj);
    renderer.modelviewmatrix(mv);

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
      case 't':
        o->trimming(!o->trimming());
        std::cout << "Trimming set to " << o->trimming() << std::endl;
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

void glfw_mousebutton(GLFWwindow* window, int button, int action, int mods)
{
  glfwMakeContextCurrent(window);

  the_app->mouse((button == GLFW_MOUSE_BUTTON_LEFT) ? gpucast::gl::eventhandler::left : gpucast::gl::eventhandler::right,
    (action == GLFW_RELEASE) ? gpucast::gl::eventhandler::release : gpucast::gl::eventhandler::press, the_app->posx(), the_app->posy());
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