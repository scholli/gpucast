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
#include <memory>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// local includes
#include <gpucast/gl/program.hpp>
#include <gpucast/gl/util/vsync.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/primitives/cube.hpp>
#include <gpucast/gl/util/trackball.hpp>
#include <gpucast/math/matrix4x4.hpp>



class application
{
public:

  application()
    : _program(),
      _cube(),
      _trackball(new gpucast::gl::trackball)
  {}

  ~application()
  {}

  void initialize()
  {
    glEnable(GL_DEPTH_TEST);

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


  void draw()
  {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    gpucast::math::matrix4f view = gpucast::math::lookat(0.0f, 0.0f, 10.0f,
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f);

    gpucast::math::matrix4f model = gpucast::math::make_translation(_trackball->shiftx(), _trackball->shifty(), _trackball->distance()) *
      _trackball->rotation();

    gpucast::math::matrix4f proj = gpucast::math::perspective(60.0f, 1.0f, 1.0f, 1000.0f);
    gpucast::math::matrix4f mv = view * model;
    gpucast::math::matrix4f mvp = proj * mv;
    gpucast::math::matrix4f nm = mv.normalmatrix();

    _program->begin();

    _program->set_uniform_matrix4fv("modelviewprojectionmatrix", 1, false, &mvp[0]);
    _program->set_uniform_matrix4fv("modelviewmatrix", 1, false, &mv[0]);
    _program->set_uniform_matrix4fv("normalmatrix", 1, false, &nm[0]);

    _cube->draw();

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
    case GLFW_MOUSE_BUTTON_2: button = gpucast::gl::eventhandler::middle; break;
    case GLFW_MOUSE_BUTTON_3: button = gpucast::gl::eventhandler::right; break;
    };

    _trackball->mouse(button, state, _currentx, _currenty);
  }

  void motion(double x, double y)
  {
    _currentx = int(x);
    _currenty = int(y);

    _trackball->motion(x, y);
  }

  void keyboard(int key, int action)
  {}

  void reset()
  {}

public :

  int _currentx;
  int _currenty;

  std::shared_ptr<gpucast::gl::program>   _program;
  std::shared_ptr<gpucast::gl::cube>      _cube;
  std::shared_ptr<gpucast::gl::trackball> _trackball;
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
  }

  if (the_app)
    the_app->keyboard(key, action);
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

  glfwSetFramebufferSizeCallback(window, glfw_resize);

  glfwSetKeyCallback(window, glfw_key);
  glfwSetMouseButtonCallback(window, glfw_mousebutton);
  glfwSetCursorPosCallback(window, glfw_motion);

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  glewExperimental = true;
  glewInit();

  the_app = new application;
  the_app->initialize();

  // somehow initial resize necessary?!?
  glfw_resize(window, winx, winy);

  gpucast::gl::set_vsync(false);

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
