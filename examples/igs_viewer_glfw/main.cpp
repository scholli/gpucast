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
#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/util/timer_guard.hpp>

#include <gpucast/gl/error.hpp>
#include <gpucast/math/vec3.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/gl/primitives/bezierobject.hpp>
#include <gpucast/gl/primitives/bezierobject_renderer.hpp>

#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>

#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

class application : public gpucast::gl::trackball
{
private:

  int                                               _argc;
  char**                                            _argv;

  int                                               _width;
  int                                               _height;
  unsigned                                          _frames = 0;

  typedef std::shared_ptr<gpucast::gl::bezierobject> surfaceobject_ptr;
  std::vector<surfaceobject_ptr>                    _objects;

  gpucast::math::bbox3f                             _bbox;

  std::shared_ptr<gpucast::gl::plane>               _quad;
  bool                                              _fxaa = false;
  std::shared_ptr<gpucast::gl::program>             _fxaa_program;

  std::shared_ptr<gpucast::gl::bezierobject_renderer> _renderer;

  std::shared_ptr<gpucast::gl::sampler>             _sample_linear;
  std::shared_ptr<gpucast::gl::texture2d>           _depthattachment;
  std::shared_ptr<gpucast::gl::texture2d>           _colorattachment;

public:

  application(int argc, char** argv, int width, int height)
    : _argc(argc),
    _argv(argv),
    _width(width),
    _height(height)
  {
    initialize();
  }

  ~application()
  {}


  void initialize()
  {
    //////////////////////////////////////////////////
    // scene and renderer
    //////////////////////////////////////////////////
    _renderer = std::make_shared<gpucast::gl::bezierobject_renderer>();
    std::vector<std::string> filenames;

    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/body.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/body_ground.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/rims.igs.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/top.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/small_parts.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/pipes.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/pumper.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/inner_rims.igs");
    //filenames.push_back("D:/phd/cad_data/engine-for-my-1100-cc-dissembled/inner_stuff.igs");
    filenames.push_back("C:/repositories/source/github-scholli-gpucast/examples/igs_viewer_glfw/data/vw02.txt");

    gpucast::gl::material default_material;
    default_material.ambient = gpucast::math::vec3f(0.0, 0.0, 0.0);
    default_material.diffuse = gpucast::math::vec3f(0.8, 0.8, 0.8);
    default_material.specular = gpucast::math::vec3f(0.4, 0.4, 0.4);
    default_material.opacity = 0.2;
    default_material.shininess = 1.0;

    _renderer->spheremap("D:/media/pics/hdr/DH216SN.hdr");
    _renderer->spheremapping(true);

    //////////////////////////////////////////////////
    // GL ressources
    //////////////////////////////////////////////////
    
    _renderer->add_search_path("../../../");
    _renderer->add_search_path("../../");

    gpucast::gl::resource_factory program_factory;

    _fxaa_program = program_factory.create_program({
      { gpucast::gl::vertex_stage,   "resources/glsl/base/render_from_texture.vert" },
      { gpucast::gl::fragment_stage, "resources/glsl/base/render_from_texture.frag" }
    });

    _renderer->recompile();

    _quad.reset(new gpucast::gl::plane(0, -1, 1));

    _colorattachment.reset(new gpucast::gl::texture2d);
    _depthattachment.reset(new gpucast::gl::texture2d);

    _colorattachment->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);
    _depthattachment->teximage(0, GL_DEPTH32F_STENCIL8, GLsizei(_width), GLsizei(_height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    _renderer->attach_custom_textures(_colorattachment, _depthattachment);
    _renderer->set_resolution(_width, _height);

    _sample_linear.reset(new gpucast::gl::sampler);
    _sample_linear->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
    _sample_linear->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
    _sample_linear->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _sample_linear->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //////////////////////////////////////////////////
    // GL state setup
    //////////////////////////////////////////////////
    _renderer->set_background(gpucast::math::vec3f(0.2f, 0.2f, 0.2f));
    glEnable(GL_DEPTH_TEST);

    //////////////////////////////////////////////////
    // model setup
    //////////////////////////////////////////////////
    if (_argc > 1)
    {
      filenames.clear();
      for (int i = 1; i < _argc; ++i)
      {
        filenames.push_back(_argv[i]);
      }
    }
    
    for (auto const& file : filenames)
    {
      auto ext = boost::filesystem::extension(file);
      if (ext == ".igs" || ext == ".iges" || ext == ".IGS" || ext == ".IGES") {
        open_igs(file, default_material);
      }
      if (ext == ".txt") {
        if (boost::filesystem::exists(file)) {
          std::fstream fstr(file);
          while (fstr) {
            std::string line;
            std::getline(fstr, line);

            boost::char_separator<char> sep(";");
            boost::tokenizer< boost::char_separator<char> > tokens(line, sep);
            std::vector<std::string> stokens(tokens.begin(), tokens.end());
            
            unsigned i = 0;
            if (stokens.size() == 12) {
              std::string filename = stokens[i++];
              gpucast::gl::material custom_material;
              float ar = std::stof(stokens[i++]);
              float ag = std::stof(stokens[i++]);
              float ab = std::stof(stokens[i++]);

              float dr = std::stof(stokens[i++]);
              float dg = std::stof(stokens[i++]);
              float db = std::stof(stokens[i++]);

              float sr = std::stof(stokens[i++]);
              float sg = std::stof(stokens[i++]);
              float sb = std::stof(stokens[i++]);

              custom_material.ambient   = gpucast::math::vec4f{ ar, ag, ab, 1.0f };
              custom_material.diffuse = gpucast::math::vec4f{ dr, dg, db, 1.0f };
              custom_material.specular = gpucast::math::vec4f{ sr, sg, sb, 1.0f };

              custom_material.shininess = std::stof(stokens[i++]);
              custom_material.opacity   = std::stof(stokens[i++]);

              open_igs(filename, custom_material);
            }
          }
          fstr.close();
        }
      }
    }
    reset();
  }

  void open_igs(std::string const& file, gpucast::gl::material const& m)
  {
    gpucast::igs_loader loader;
    gpucast::surface_converter converter;

    bool initialized_bbox = false;

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

      auto drawable = std::make_shared<gpucast::gl::bezierobject>(*bezier_object);
      drawable->set_material(m);
      drawable->rendermode(gpucast::gl::bezierobject::tesselation);
      drawable->trimming(gpucast::beziersurfaceobject::contour_kd_partition);

      _objects.push_back(drawable);
    }
  }


  void draw()
  {
    ++_frames;
    gpucast::gl::timer_guard full_frame("Frame total");

    float near_clip = 0.01f * _bbox.size().abs();
    float far_clip  = 2.0f  * _bbox.size().abs();

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, float(_bbox.size().abs()),
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::vec3f translation = _bbox.center();

    gpucast::math::matrix4f model = gpucast::math::make_translation(shiftx(), shifty(), distance()) * rotation() *
      gpucast::math::make_translation(-translation[0], -translation[1], -translation[2]);

    gpucast::math::matrix4f proj = gpucast::math::perspective(50.0f, float(_width) / _height, near_clip, far_clip);

    auto mvp = proj * view * model;
    auto mvpi = gpucast::math::inverse(mvp);

    _renderer->set_nearfar(near_clip, far_clip);
    _renderer->set_resolution(_width, _height);
    _renderer->view_setup(view, model, proj);

    {
      //gpucast::gl::timer_guard t("renderer->begin_draw()");
      _renderer->begin_draw(); 
    }

    for (auto const& o : _objects)
    {
      //gpucast::gl::timer_guard t("Draw object");
      if (_renderer->inside_frustum(*o)) {
        o->draw(*_renderer);
      }
    }

    {
      //gpucast::gl::timer_guard t("renderer->end_draw()");
      _renderer->end_draw();
    }

    {
      gpucast::gl::timer_guard t("fxaa blit");
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glClearColor(0.4, 0.0, 0.0, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

      _fxaa_program->begin();

      _fxaa_program->set_uniform_matrix4fv("modelviewprojectioninverse", 1, false, &mvpi[0]);
      _fxaa_program->set_uniform_matrix4fv("modelviewprojection", 1, false, &mvp[0]);

      _fxaa_program->set_texture2d("colorbuffer", *_colorattachment, 1);
      _fxaa_program->set_texture2d("depthbuffer", *_depthattachment, 2);

      _sample_linear->bind(1);
      _sample_linear->bind(2);

      _fxaa_program->set_uniform1i("fxaa_mode", int(_fxaa));
      _fxaa_program->set_uniform1i("width", GLsizei(_width));
      _fxaa_program->set_uniform1i("height", GLsizei(_height));
      _quad->draw();

      _fxaa_program->end();
    }
  }

  void resize(int w, int h)
  {
    _width = w; _height = h;
    glViewport(0, 0, GLsizei(_width), GLsizei(_height));

    _colorattachment->teximage(0, GL_RGBA32F, GLsizei(_width), GLsizei(_height), 0, GL_RGBA, GL_FLOAT, 0);
    _depthattachment->teximage(0, GL_DEPTH32F_STENCIL8, GLsizei(_width), GLsizei(_height), 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    _renderer->attach_custom_textures(_colorattachment, _depthattachment);
    _renderer->set_resolution(_width, _height);
  }

public:

  virtual void keyboard(unsigned char key, int x, int y) override
  {
    // renderer operations
    switch (key)
    {
    case 'c':
    case 'C':
      _renderer->recompile();
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
        o->raycasting_max_iterations(std::max(1U, o->raycasting_max_iterations() - 1));
        std::cout << "Newton iterations set to " << o->raycasting_max_iterations() << std::endl;
        break;
      case 'I':
        o->raycasting_max_iterations(o->raycasting_max_iterations() + 1);
        std::cout << "Newton iterations set to " << o->raycasting_max_iterations() << std::endl;
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
  int winx = 1920;
  int winy = 1080;

  glfwInit();

  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);

  GLFWwindow* window = glfwCreateWindow(winx, winy, "IGS View GLFW", NULL, NULL);
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

  the_app = std::make_shared<application>(argc, argv, winx, winy);

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