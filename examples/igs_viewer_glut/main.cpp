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
#include <gpucast/gl/util/contextinfo.hpp>
#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/util/init_glew.hpp>

#include <gpucast/gl/atomicbuffer.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/timer_query.hpp>
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>
#include <gpucast/glut/window.hpp>

#include <gpucast/math/matrix4x4.hpp> 
#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp> 

#include <gpucast/gl/primitives/bezierobject.hpp>
#include <gpucast/gl/primitives/cube.hpp>

#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>

static unsigned const initial_window_width = 1920;
static unsigned const initial_window_height = 1080;

class application : public gpucast::gl::trackball
{
private : 

  int                                               _argc;
  char**                                            _argv;

  typedef std::shared_ptr<gpucast::gl::bezierobject> surfaceobject_ptr;
  std::vector<surfaceobject_ptr>                    _objects;

  std::vector<std::shared_ptr<gpucast::gl::cube>>   _obbs;
  std::shared_ptr<gpucast::gl::program>             _program;
  bool                                              _show_obbs = true;

  gpucast::gl::timer_query                          _query;
  gpucast::math::bbox3f                             _bbox;

  unsigned                                          _resolution_x = initial_window_width;
  unsigned                                          _resolution_y = initial_window_height;

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
    glEnable(GL_CULL_FACE);

    gpucast::igs_loader loader;
    gpucast::surface_converter converter;

    auto renderer = gpucast::gl::bezierobject_renderer::instance();

    renderer->add_search_path("../../../../");
    renderer->add_search_path("../../../");
    renderer->add_search_path("../../");
    renderer->add_search_path("../");
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
    }

    for (auto const& file : filenames)
    {
      std::cout << "loading " << file << std::endl;
       
#if 1
      gpucast::igs_loader loader2;
      auto nurbs_objects = loader2.load(file);
#else
      auto nurbs_objects = loader.load(file);
#endif

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
        std::cout << "Bbox : " << bezier_object->bbox().min << " - " << bezier_object->bbox().max << std::endl;
        
        for (auto i = bezier_object->begin(); i != bezier_object->end(); ++i) {
          auto obb_gl = std::make_shared<gpucast::gl::cube>();
          _obbs.push_back(obb_gl);

          gpucast::math::obbox3d obb((**i).mesh(), gpucast::math::partial_derivative_policy<gpucast::math::point3d>());

          auto phigh = obb.high();
          auto plow = obb.low();

          auto orientation = obb.orientation();
          auto inv_orientation = compute_inverse(orientation);

          auto lbf = orientation * gpucast::math::point3d(plow[0], plow[1], plow[2])    + obb.center();  // left, bottom, front
          auto rbf = orientation * gpucast::math::point3d(phigh[0], plow[1], plow[2])   + obb.center();  // right, bottom, front
          auto rtf = orientation * gpucast::math::point3d(phigh[0], phigh[1], plow[2])  + obb.center();  // right, top, front
          auto ltf = orientation * gpucast::math::point3d(plow[0], phigh [1], plow[2])  + obb.center();  // left, top, front

          auto lbb = orientation * gpucast::math::point3d(plow[0], plow[1], phigh[2])   + obb.center(); // left, bottom, back  
          auto rbb = orientation * gpucast::math::point3d(phigh[0], plow[1], phigh[2])  + obb.center(); // right, bottom, back  
          auto rtb = orientation * gpucast::math::point3d(phigh[0], phigh[1], phigh[2]) + obb.center(); // right, top, back  
          auto ltb = orientation * gpucast::math::point3d(plow[0], phigh[1], phigh[2])  + obb.center(); // left, top, back  

          lbf.weight(1.0);
          rbf.weight(1.0);
          rtf.weight(1.0);
          ltf.weight(1.0);
          lbb.weight(1.0);
          rbb.weight(1.0);
          rtb.weight(1.0);
          ltb.weight(1.0);

          obb_gl->set_vertices(lbf, rbf,ltf, rtf, lbb, rbb, ltb, rtb);
          obb_gl->set_colors(gpucast::math::vec4f{ 0.0f, 0.0f, 0.0f, 1.0 }, 
                             gpucast::math::vec4f{ 1.0f, 0.0f, 0.0f, 1.0 }, 
                             gpucast::math::vec4f{ 0.0f, 1.0f, 0.0f, 1.0 }, 
                             gpucast::math::vec4f{ 1.0f, 1.0f, 0.0f, 1.0 }, 
                             gpucast::math::vec4f{ 0.0f, 0.0f, 1.0f, 1.0 }, 
                             gpucast::math::vec4f{ 1.0f, 0.0f, 1.0f, 1.0 }, 
                             gpucast::math::vec4f{ 0.0f, 1.0f, 1.0f, 1.0 }, 
                             gpucast::math::vec4f{ 1.0f, 1.0f, 1.0f, 1.0 });
        }

        gpucast::gl::material mat;
        mat.randomize(0.05f, 1.0f, 0.1f, 20.0f, 1.0f);

        auto drawable = std::make_shared<gpucast::gl::bezierobject>(*bezier_object);
        drawable->set_material(mat);
        drawable->trimming(gpucast::beziersurfaceobject::contour_kd_partition);

        _objects.push_back(drawable);
      }
    }

    // build simple bbox program
    _program = std::make_shared<gpucast::gl::program>();

    std::string vtx_str = R"(
      #version 440 core
      #extension GL_NV_gpu_shader5 : enable

      layout (location = 0) in vec4 in_position;
      layout (location = 1) in vec4 in_color;

      uniform mat4 mvp;

      out vec4 color;

      void main() {
        gl_Position = mvp * in_position;
        color = in_color;
      }    
    )";

    std::string frg_str = R"(
      #version 440 core
      #extension GL_NV_gpu_shader5 : enable

      in vec4 color;

      layout(location = 0) out vec3 out_color;

      void main() {
        out_color = color.xyz;
      }    
    )";

    gpucast::gl::shader vertex_shader(gpucast::gl::vertex_stage);
    vertex_shader.set_source(vtx_str.c_str());
    std::cout << "Vertex shader log" << vertex_shader.log() << std::endl;

    gpucast::gl::shader fragment_shader(gpucast::gl::fragment_stage);
    fragment_shader.set_source(frg_str.c_str());
    std::cout << "Fragment shader log : " << fragment_shader.log() << std::endl;

    _program->add(&vertex_shader);
    _program->add(&fragment_shader);
    _program->link();

    std::cout << "Program log : " << _program->log() << std::endl;
  }


  void draw()
  {
    _query.begin();

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    auto renderer = gpucast::gl::bezierobject_renderer::instance();

    float nearclip = 0.01f * _bbox.size().abs();
    float farclip = 5.0f * _bbox.size().abs();

    renderer->set_nearfar(nearclip, farclip);
    renderer->set_resolution(_resolution_x, _resolution_y);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, float(_bbox.size().abs()),
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::vec3f translation = _bbox.center();

    gpucast::math::matrix4f model = gpucast::math::make_translation(shiftx(), shifty(), distance()) * rotation() *
                                    gpucast::math::make_translation(-translation[0], -translation[1], -translation[2]);

    gpucast::math::matrix4f proj = gpucast::math::perspective(30.0f, float(_resolution_x) / float(_resolution_y), nearclip, farclip);

    renderer->current_modelmatrix(model);
    renderer->current_viewmatrix(view);
    renderer->current_projectionmatrix(proj);

    for (auto const& o : _objects) {
      if (o->enable_raycasting()) {
        o->draw(gpucast::gl::bezierobject::raycasting);
      }
      else {
        o->draw(gpucast::gl::bezierobject::tesselation);
      }
    }

    _query.end();

    std::cout << "fps: " << double(1000) / _query.time_in_ms() << "\r";
#if 1

    if (_show_obbs) {
      gpucast::math::matrix4f mv = view * model;
      gpucast::math::matrix4f mvp = proj * mv;

      _program->begin();
      _program->set_uniform_matrix4fv("mvp", 1, false, &mvp[0]);

      for (auto const& o : _obbs) {

        o->draw(true);
      }
      _program->end();
    }
#endif
  }

  void resize(int w, int h)
  {
    glViewport(0, 0, w, h);
    _resolution_x = 1920;
    _resolution_y = 1080;
  }

public:

  virtual void keyboard(unsigned char key, int x, int y) override
  {
    auto renderer = gpucast::gl::bezierobject_renderer::instance();

    // renderer operations
    switch (key)
    {
    case 'c':
      renderer->recompile();
      std::cout << "Shaders recompiled" << std::endl;
      break;
    }

    // per object operations
    for (auto const& o : _objects)
    {
      switch (key)
      {
      case 'c':
        renderer->recompile();
        std::cout << "Recompiling shaders..." << std::endl;
        break;
      case 'w':
        switch (o->fillmode()) {
        case gpucast::gl::bezierobject::FILL_SOLID : 
          o->fillmode(gpucast::gl::bezierobject::FILL_WIREFRAME); 
          break;
        case gpucast::gl::bezierobject::FILL_WIREFRAME : 
          o->fillmode(gpucast::gl::bezierobject::FILL_POINT);
          break;
        case gpucast::gl::bezierobject::FILL_POINT : 
          o->fillmode(gpucast::gl::bezierobject::FILL_SOLID);
          break;
        }
        std::cout << "Recompiling shaders..." << std::endl;
        break;
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
      case 'o':
        _show_obbs = !_show_obbs;
        std::cout << "Rendering bbox set to " << _show_obbs << std::endl;
        break;
      }
    }
  }
};

int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, 1920, 1080, 100, 100, 4, 4, true);
  auto& win = gpucast::gl::glutwindow::instance();

  gpucast::gl::init_glew(std::cout);

  gpucast::gl::print_contextinfo(std::cout);

  auto the_app = std::make_shared<application>(argc, argv);

  win.add_eventhandler(the_app);
  
  auto draw_fun = std::make_shared<std::function<void()>>(std::bind(&application::draw, the_app));
  win.set_drawfunction(draw_fun);

  win.run();

  return 0;
}

