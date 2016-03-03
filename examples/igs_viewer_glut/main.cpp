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
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/glut/window.hpp>

#include <gpucast/math/matrix4x4.hpp> 
#include <gpucast/math/oriented_boundingbox_partial_derivative_policy.hpp> 

#include <gpucast/gl/primitives/bezierobject.hpp>
#include <gpucast/gl/primitives/cube.hpp>

#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>

static const unsigned resolution_x = 1920;
static const unsigned resolution_y = 1080;

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
          auto ltf = orientation * gpucast::math::point3d(plow[0], plow[1], plow[2])    + obb.center();  // left, top, front
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
        }

        gpucast::gl::material mat;
        mat.randomize(0.05f, 1.0f, 0.1f, 20.0f, 1.0f);

        auto drawable = std::make_shared<gpucast::gl::bezierobject>(*bezier_object);
        drawable->set_material(mat);
        drawable->trimming(gpucast::beziersurfaceobject::contour_kd_partition);

        _objects.push_back(drawable);
      }
      else {
        std::cerr << "failed to load " << file << std::endl;
        std::cerr << loader.error_message() << std::endl;
      }
    }

    // build simple bbox program
    _program = std::make_shared<gpucast::gl::program>();

    std::string vtx_str = R"(
      #version 440 core
      #extension GL_NV_gpu_shader5 : enable

      layout (location = 0) in vec4 in_position;

      uniform mat4 mvp;

      void main() {
        gl_Position = mvp * in_position;
      }    
    )";

    std::string frg_str = R"(
      #version 440 core
      #extension GL_NV_gpu_shader5 : enable

      layout(location = 0) out vec3 out_color;

      void main() {
        out_color = vec3(1);
      }    
    )";

    gpucast::gl::vertexshader vertex_shader;
    vertex_shader.set_source(vtx_str.c_str());
    vertex_shader.compile();
    std::cout << "Vertex shader log" << vertex_shader.log() << std::endl;

    gpucast::gl::fragmentshader fragment_shader;
    fragment_shader.set_source(frg_str.c_str());
    fragment_shader.compile();
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

    auto& renderer = gpucast::gl::bezierobject_renderer::instance();

    float nearclip = 0.01f * _bbox.size().abs();
    float farclip = 3.0f * _bbox.size().abs();

    renderer.set_nearfar(nearclip, farclip);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, float(_bbox.size().abs()),
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::vec3f translation = _bbox.center();

    gpucast::math::matrix4f model = gpucast::math::make_translation(shiftx(), shifty(), distance()) * rotation() *
                                    gpucast::math::make_translation(-translation[0], -translation[1], -translation[2]);

    gpucast::math::matrix4f proj = gpucast::math::perspective(30.0f, float(resolution_x) / float(resolution_y), nearclip, farclip);
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

    if (_show_obbs) {

      _program->begin();
      _program->set_uniform_matrix4fv("mvp", 1, false, &mvp[0]);

      for (auto const& o : _obbs) {

        o->draw(true);
      }
      _program->end();
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
      case 'o':
        _show_obbs = !_show_obbs;
        break;
      }
    }
  }
};

int main(int argc, char** argv)
{
  gpucast::gl::glutwindow::init(argc, argv, resolution_x, resolution_y, 100, 100, 4, 3, true);
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
