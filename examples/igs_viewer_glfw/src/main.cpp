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

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// local includes
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/fragmentshader.hpp>

#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/surface_renderer_gl.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>


class application
{
public:

  application()
    : _trackball(new gpucast::gl::trackball),
      _renderer(),
      _cull_backface(true)
  {}

  ~application()
  {}

  void init_program()
  {
    std::string vertexshader_code = R"(
    #version 420 core
      #extension GL_ARB_separate_shader_objects : enable 
                 
      layout (location = 0) in vec4 vertex;   
      layout (location = 1) in vec4 texcoord;   
      layout (location = 2) in vec4 normal;   
                                         
      uniform mat4 modelviewprojectionmatrix; 
      uniform mat4 modelviewmatrix; 
      uniform mat4 normalmatrix; 
                                                                 
      out vec4 fragnormal;  
      out vec4 fragtexcoord;
      out vec4 fragposition;
                                                                                         
      void main(void) 
      { 
        fragtexcoord = texcoord; 
        fragnormal   = normalmatrix * normal; 
        fragposition = modelviewmatrix * vertex; 
        gl_Position  = modelviewprojectionmatrix * vertex; 
      })";

    std::string fragmentshader_code = R"(
    #version 420 core
      #extension GL_ARB_separate_shader_objects : enable 
                  
      in vec4 fragnormal;   
      in vec4 fragtexcoord; 
      in vec4 fragposition; 
                                          
      layout (location = 0) out vec4 color; 
                                                      
      void main(void) 
      { 
        vec3 V = normalize(-fragposition.xyz);  
        vec3 N = normalize(fragnormal.xyz); 
        float attenuation = min(1.0, 10.0 / length(fragposition.xyz)); 
        color = attenuation * vec4(1.0) * dot(N,V)  +  0.1 * fragtexcoord; 
      }
    )";

    gpucast::gl::vertexshader   vs;
    gpucast::gl::fragmentshader fs;

    vs.set_source(vertexshader_code.c_str());
    fs.set_source(fragmentshader_code.c_str());

    vs.compile();
    fs.compile();

    _program = std::make_shared<gpucast::gl::program>();
    _program->add(&fs);
    _program->add(&vs);

    std::cout << "vertex shader log : " << vs.log() << std::endl;
    std::cout << "fragment shader log : " << fs.log() << std::endl;

    _cube = std::make_shared<gpucast::gl::cube>(0, -1, 2, 1);

    _program->link();
  }



  void initialize(int argc, char** argv)
  {
    glEnable(GL_DEPTH_TEST);

    gpucast::igs_loader loader;
    gpucast::surface_converter converter;

    _renderer = std::make_shared<gpucast::surface_renderer_gl>(argc, argv);
    _renderer->add_path("../../../gpucast_core/glsl");
    _renderer->add_path("../../../");

    for (int i = 1; i < argc; ++i)
    {
      std::shared_ptr<gpucast::beziersurfaceobject> bezier_object = _renderer->create();
      std::shared_ptr<gpucast::nurbssurfaceobject> nurbs_object = loader.load(argv[i]);

      if (nurbs_object)
      {
        converter.convert(nurbs_object, bezier_object);
        _objects.push_back(bezier_object);
      } else {
        std::cerr << "failed to load " << argv[i] << std::endl;
        std::cerr << loader.error_message() << std::endl;

      }
    }

    if (!_objects.empty()) 
    {
      _bbox.min = _objects.front()->bbox().min;
      _bbox.max = _objects.front()->bbox().max;
    }

    for (auto o : _objects)
    {
      _bbox.merge(o->bbox().min);
      _bbox.merge(o->bbox().max);
    }

    std::cout << _bbox << std::endl;

    init_program();
    _cube->set_vertices(
      gpucast::gl::vec4f(_bbox.min[0], _bbox.min[1], _bbox.min[2], 1.0f),
      gpucast::gl::vec4f(_bbox.max[0], _bbox.min[1], _bbox.min[2], 1.0f),
      gpucast::gl::vec4f(_bbox.min[0], _bbox.max[1], _bbox.min[2], 1.0f),
      gpucast::gl::vec4f(_bbox.max[0], _bbox.max[1], _bbox.min[2], 1.0f),
      gpucast::gl::vec4f(_bbox.min[0], _bbox.min[1], _bbox.max[2], 1.0f),
      gpucast::gl::vec4f(_bbox.max[0], _bbox.min[1], _bbox.max[2], 1.0f),
      gpucast::gl::vec4f(_bbox.min[0], _bbox.max[1], _bbox.max[2], 1.0f),
      gpucast::gl::vec4f(_bbox.max[0], _bbox.max[1], _bbox.max[2], 1.0f) );
  }


  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    if (_cull_backface)
      glEnable(GL_CULL_FACE);
    else
      glDisable(GL_CULL_FACE);

    gpucast::gl::matrix4f view = gpucast::gl::lookat<float>(0.0f, 0.0f, 2.0f * _bbox.size().abs(),
                                                            0.0f, 0.0f, 0.0f,
                                                            0.0f, 1.0f, 0.0f);

    gpucast::gl::matrix4f model = gpucast::gl::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
                                  _trackball->rotation() *
                                  gpucast::gl::make_translation(-_bbox.center()[0], -_bbox.center()[1], -_bbox.center()[2]);

    gpucast::gl::matrix4f proj = gpucast::gl::perspective(60.0f, 1.0f, 1.0f, 200000.0f);
    gpucast::gl::matrix4f mv = view * model;
    gpucast::gl::matrix4f mvp = proj * mv;
    gpucast::gl::matrix4f nm = mv.normalmatrix();

    _renderer->nearplane(1.0f);
    _renderer->farplane(200000.0f);
    _renderer->modelviewmatrix(mv);
    _renderer->projectionmatrix(proj);
    
    for (auto o : _objects)
    {
      // draw here
    }

    _renderer->draw();

    _program->begin();

    _program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program->set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    _cube->draw(true);

    _program->end();
  }

  void resize(int w, int h)
  {
    glViewport(0, 0, w, h);
  }

  void mousebutton(int b, int action)
  {
    gpucast::gl::eventhandler::state state = (action == GLFW_PRESS) ? gpucast::gl::eventhandler::press : gpucast::gl::eventhandler::release;
    gpucast::gl::eventhandler::button button;

    switch (b) {
      case GLFW_MOUSE_BUTTON_1: button = gpucast::gl::eventhandler::left; break;
      case GLFW_MOUSE_BUTTON_2: button = gpucast::gl::eventhandler::right; break;
      case GLFW_MOUSE_BUTTON_3: button = gpucast::gl::eventhandler::middle; break;
    };

    _trackball->mouse(button, state, _currentx, _currenty);
  }

  void motion(double x, double y)
  {
    _currentx = int(x);
    _currenty = int(y);

    _trackball->motion(_currentx, _currenty);
  }

  void keyboard(int key, int action)
  {
    switch (key) 
    {
      case 'C':
        _renderer->recompile(); 
        std::cout << "Shaders recompiled" << std::endl;
        break;
      case 'B':
        _cull_backface = !_cull_backface;
        std::cout << "Backface culling set to " << _cull_backface << std::endl;
        break;
      case 'I':
        _renderer->newton_iterations(_renderer->newton_iterations()-1);
        std::cout << "Newton iterations set to " << _renderer->newton_iterations() << std::endl;
        break;
      case 'O':
        _renderer->newton_iterations(_renderer->newton_iterations()+1);
        std::cout << "Newton iterations set to " << _renderer->newton_iterations() << std::endl;
        break;
      case 'R':
        _renderer->raycasting(!_renderer->raycasting());
        std::cout << "Raycasting set to " << _renderer->raycasting() << std::endl;
        break;
      case 'T':
        _renderer->trimming(!_renderer->trimming());
        std::cout << "Trimming set to " << _renderer->trimming() << std::endl;
        break;
    }
  }

  void reset()
  {}

public:

  int _currentx;
  int _currenty;

  bool                                                        _cull_backface;

  gpucast::math::bbox3f                                       _bbox;
  std::shared_ptr<gpucast::gl::cube>                          _cube;
  std::shared_ptr<gpucast::gl::program>                       _program;

  std::shared_ptr<gpucast::surface_renderer_gl>               _renderer;
  std::vector<std::shared_ptr<gpucast::beziersurfaceobject>>  _objects;
  std::shared_ptr<gpucast::gl::trackball>                     _trackball;
};

application* the_app = nullptr;

void
glut_display()
{
  if (the_app)
    the_app->draw();
}

void
glfw_resize(GLFWwindow* window, int w, int h)
{
  glfwMakeContextCurrent(window);
  if (the_app)
    the_app->resize(w, h);
}

void glfw_mousebutton(GLFWwindow* window, int button, int action, int mods)
{
  glfwMakeContextCurrent(window);
  if (the_app)
    the_app->mousebutton(button, action);
}

void glfw_motion(GLFWwindow* window, double x, double y)
{
  glfwMakeContextCurrent(window);
  if (the_app)
    the_app->motion(x, y);
}

void glfw_key(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  std::cout << "Key released : " << char(key) << " action" << action << " mod: " << mods << std::endl;

  glfwMakeContextCurrent(window);
  if (action == GLFW_RELEASE)
  {
    switch (key) {
      // ESC key
    case 27:
    {
      if (the_app) {
        the_app->reset();
        std::cout << "reset application" << std::endl;
        exit(0);
      }
    }
      break;
    case 'f':
      //toggle_fullscreen();
      break;
    }

    if (the_app) {
      the_app->keyboard(key, action);
      
    }
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

  GLFWwindow* window = glfwCreateWindow(winx, winy, "igs_viewer_glfw", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  glfwSetFramebufferSizeCallback(window, glfw_resize);

  glfwSetKeyCallback(window, glfw_key);
  glfwSetMouseButtonCallback(window, glfw_mousebutton);
  glfwSetCursorPosCallback(window, glfw_motion);

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  glewExperimental = true;
  glewInit();

  the_app = new application;
  the_app->initialize(argc, argv);

  // somehow initial resize necessary?!?
  glfw_resize(window, winx, winy);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
    glut_display();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  delete the_app;

  return 0;
}
